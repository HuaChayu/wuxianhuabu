//
//  Safetensors-mmap映射-通用.swift
//  无限画布 — safetensors 只读内存映射（避免多复制一份）
//
//  ============================================================
//  为什么需要它：
//    FileHandle.read(upToCount:) → Data → MLXArray(data:) 的写法，
//    每个 tensor 会先落一块 Data 堆副本，再复制进 MLX 的 unified memory，
//    全程两份。这里改成 open + mmap(MAP_PRIVATE) 只读映射：
//      · 进程不再持有整块匿名内存副本，页来自文件页缓存（clean page），
//        内存压力下内核可直接回收；
//      · 取 tensor 时用 Data(bytesNoCopy:) 指向映射区，只做一次
//        MLXArray 物化（GPU 必须能读到这块 buffer，这一步省不掉）。
//
//  与 MLX 自带 loadArrays(url:) 的关系：
//    MLX C++ 侧 ParallelFileReader 是 pread 直接读进 array 自己的 buffer，
//    也只有一份。凡走 loadArrays / loadSafetensors 的调用（LTX / HiDream /
//    ZImage）本来就不存在多余复制，无需改造；本文件只服务于自己手写解析
//    header 的加载器（H3Weights、readSafetensorsFloatArrays 等）。
//  ============================================================

import Foundation
import MLX

enum MappedSafetensorsError: Error, LocalizedError {
    case openFailed(String)
    case mmapFailed(String)
    case badHeader(String)

    var errorDescription: String? {
        switch self {
        case .openFailed(let s): return "open failed: \(s)"
        case .mmapFailed(let s): return "mmap failed: \(s)"
        case .badHeader(let s): return "bad header: \(s)"
        }
    }
}

/// safetensors 文件的只读映射视图。header 解析一次，张量按需就地读取。
final class MappedSafetensors {
    struct Entry {
        let dtype: String      // "F32" / "BF16" / "F16" / "U32" ...
        let shape: [Int]
        let offset: Int        // 相对文件起始的绝对字节偏移
        let nbytes: Int
    }

    private let fd: Int32
    private let base: UnsafeMutableRawPointer
    private let size: Int

    let entries: [String: Entry]
    /// 张量数据区起始（8 + header 长度）
    let dataStart: Int

    init(url: URL) throws {
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw MappedSafetensorsError.openFailed(url.lastPathComponent)
        }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 16 else {
            close(fd)
            throw MappedSafetensorsError.badHeader("file too small: \(url.lastPathComponent)")
        }
        let size = Int(st.st_size)
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), base != MAP_FAILED else {
            close(fd)
            throw MappedSafetensorsError.mmapFailed(url.lastPathComponent)
        }

        let headerLen = base.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        guard headerLen > 0, Int(headerLen) + 8 <= size else {
            munmap(base, size); close(fd)
            throw MappedSafetensorsError.badHeader("header length out of range: \(url.lastPathComponent)")
        }
        let headerData = Data(bytes: base + 8, count: Int(headerLen))
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            munmap(base, size); close(fd)
            throw MappedSafetensorsError.badHeader("header is not JSON: \(url.lastPathComponent)")
        }

        let start = 8 + Int(headerLen)
        var parsed: [String: Entry] = [:]
        parsed.reserveCapacity(obj.count)
        for (key, value) in obj {
            guard key != "__metadata__", let t = value as? [String: Any],
                  let dtypeName = t["dtype"] as? String,
                  let shape = t["shape"] as? [NSNumber],
                  let offs = t["data_offsets"] as? [NSNumber], offs.count == 2 else { continue }
            let lo = offs[0].intValue, hi = offs[1].intValue
            guard hi >= lo, start + hi <= size else { continue }
            parsed[key] = Entry(
                dtype: dtypeName,
                shape: shape.map { $0.intValue },
                offset: start + lo,
                nbytes: hi - lo
            )
        }

        self.fd = fd
        self.base = base
        self.size = size
        self.dataStart = start
        self.entries = parsed
    }

    deinit {
        munmap(base, size)
        close(fd)
    }

    /// 指向映射区数据的裸指针（不复制）。
    func pointer(_ key: String) -> UnsafeRawPointer? {
        guard let e = entries[key] else { return nil }
        guard e.offset >= 0, e.offset + e.nbytes <= size else { return nil }
        return UnsafeRawPointer(base + e.offset)
    }

    /// 零拷贝 Data（deallocator 为 .none，生命周期由本对象保证）。
    func bytes(_ key: String, copy: Bool) -> Data? {
        guard let e = entries[key] else { return nil }
        guard e.offset >= 0, e.offset + e.nbytes <= size else { return nil }
        if copy {
            return Data(bytes: base + e.offset, count: e.nbytes)
        }
        return Data(bytesNoCopy: base + e.offset, count: e.nbytes, deallocator: .none)
    }

    /// 就地物化成一个 MLXArray（只有这一份进 unified memory）。
    func mlxArray(_ key: String, dtype: MLX.DType) -> MLXArray? {
        guard let e = entries[key], let d = bytes(key, copy: false) else { return nil }
        return MLXArray(d, e.shape, dtype: dtype)
    }
}
