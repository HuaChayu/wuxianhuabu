//
//  H3ContinuationCache.swift
//  无限画布
//
//  H3 视频节点「尾帧延续」缓存容器（.h3cc）。
//
//  单文件容器布局：
//    magic "H3CC"（4B）+ headerLen UInt64 LE（8B）+ JSON header（UTF-8）+ fp16 raw video latent payload
//
//  只落盘末尾 2~4 秒窗口的 video latent（不存全量、不存 KV cache）。
//  提供 save / load / delete 三个入口；delete 幂等；load 做严格 fingerprint 校验
//  （模型 / 分辨率 / steps / latentT / 帧数 / 参考图数任一变化即拒绝复用）。
//  文件命名：<nodeID>[.-<branchID>].h3cc，存储于 <canvasRoot>/cache/h3-continuation/。
//

import Foundation
import MLX

public enum H3ContinuationCache {

    // MARK: - 头部与载荷结构

    /// .h3cc 文件头（JSON 段），含完整 fingerprint。
    public struct Header: Codable, Equatable {
        public var version: Int = 2
        public var nodeID: UUID
        public var branchID: String?
        public var winSeconds: Double           // 窗口秒数（2~4 区间）
        public var winFrames: UInt32            // 对齐后窗口帧数
        public var winLatentT: UInt32           // 窗口 latent 帧数
        public var winLatentRows: Int           // 窗口行数 = winLatentT * gridH * gridW
        public var winAudioT: UInt32            // 窗口 audio latent 帧数
        // ── v2 条件 token 透传（直接注入方案）──
        // 本节点干净 condRows（不加 aug noise）落盘，供下游节点与自身条件直接拼接注入。
        // nil/0 = 旧版本或无条件行。
        public var condRowsRows: Int?
        public var condRowsVPatch: Int?
        // ── v2 资源去重（id 透传）──
        // condRows 按资源顺序拼接，每段对应一个资源 id（路径）与行数；
        // 下游注入时按自身资源 id 匹配，命中则跳过该段，避免重复注入同一资源。
        public var condIDs: [String] = []
        public var condRowSpans: [Int] = []

        // ── fingerprint（任一变化即拒绝复用）──
        public var modelKey: String             // 模型目录名（MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit）
        public var width: Int
        public var height: Int
        public var steps: UInt32
        public var latentT: UInt32              // 源节点总 latent 帧（对齐后）
        public var frameCount: UInt32           // 源节点对齐后总帧
        public var refCount: Int                // 参考图数
        public var latH: Int
        public var latW: Int
        public var latC: Int
        public var gridH: Int
        public var gridW: Int
        public var videoPatchDim: Int
        public var createdAt: TimeInterval

        public init(nodeID: UUID, branchID: String?, winSeconds: Double,
                    winFrames: UInt32, winLatentT: UInt32, winLatentRows: Int, winAudioT: UInt32,
                    modelKey: String, width: Int, height: Int, steps: UInt32,
                    latentT: UInt32, frameCount: UInt32, refCount: Int,
                    latH: Int, latW: Int, latC: Int, gridH: Int, gridW: Int,
                    videoPatchDim: Int, createdAt: TimeInterval = Date().timeIntervalSince1970,
                    condRowsRows: Int? = nil, condRowsVPatch: Int? = nil,
                    condIDs: [String] = [], condRowSpans: [Int] = []) {
            self.nodeID = nodeID
            self.branchID = branchID
            self.winSeconds = winSeconds
            self.winFrames = winFrames
            self.winLatentT = winLatentT
            self.winLatentRows = winLatentRows
            self.winAudioT = winAudioT
            self.modelKey = modelKey
            self.width = width
            self.height = height
            self.steps = steps
            self.latentT = latentT
            self.frameCount = frameCount
            self.refCount = refCount
            self.latH = latH
            self.latW = latW
            self.latC = latC
            self.gridH = gridH
            self.gridW = gridW
            self.videoPatchDim = videoPatchDim
            self.createdAt = createdAt
            self.condRowsRows = condRowsRows
            self.condRowsVPatch = condRowsVPatch
            self.condIDs = condIDs
            self.condRowSpans = condRowSpans
        }
    }

    /// warm-up 短重采样参数（读回缓存后重建上下文用）。
    public struct Recovery {
        public var warmStart: Double
        public var warmSteps: Int
        public init(warmStart: Double = 0.85, warmSteps: Int = 1) {
            self.warmStart = warmStart
            self.warmSteps = warmSteps
        }
    }

    /// load 成功返回的三元组：头部 + fp16 raw latent + warm-up 参数。
    public struct Loaded {
        public var header: Header
        public var latentData: Data          // fp16 raw [winLatentT, H, W, C] 连续
        public var recovery: Recovery
        /// v2+：本节点干净 condRows（fp16 [condRowsRows, condRowsVPatch]），nil = 旧版本或无条件行。
        public var condRowsData: Data?
    }

    /// 消费侧期望的 fingerprint（由 generateVideo 实时参数构造）。
    public struct Expect {
        public var modelKey: String
        public var width: Int
        public var height: Int
        public var steps: UInt32
        public var latentT: UInt32
        public var frameCount: UInt32
        public var refCount: Int

        public init(modelKey: String, width: Int, height: Int, steps: UInt32,
                    latentT: UInt32, frameCount: UInt32, refCount: Int) {
            self.modelKey = modelKey
            self.width = width
            self.height = height
            self.steps = steps
            self.latentT = latentT
            self.frameCount = frameCount
            self.refCount = refCount
        }
    }

    /// 窗口几何（由秒数推导，2~4 秒区间）。
    public struct Window {
        public var seconds: Double
        public var frames: UInt32
        public var latentT: UInt32
        public var audioT: UInt32
        public func rows(gridH: Int, gridW: Int) -> Int { Int(latentT) * gridH * gridW }
    }

    public static let currentModelKey = "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit"
    public static let fileExtension = "h3cc"

    private static let magic: [UInt8] = [0x48, 0x33, 0x43, 0x43]   // "H3CC"
    private static let magicByteCount = 4
    private static let headerLenByteCount = 8

    // MARK: - 路径

    /// 缓存根目录：<canvasRoot>/cache/h3-continuation
    public static func cacheRoot(canvasRoot: String) -> String {
        canvasRoot + "/cache/h3-continuation"
    }

    public static func cacheURL(nodeID: UUID, branchID: String?, rootDir: String) -> URL {
        var name = nodeID.uuidString
        if let branchID, !branchID.isEmpty {
            name += "." + branchID
        }
        name += "." + fileExtension
        return URL(fileURLWithPath: rootDir, isDirectory: true).appendingPathComponent(name)
    }

    // MARK: - 窗口计算

    /// 由秒数计算窗口（夹取 2~4 秒，默认 3 秒）。
    public static func window(forSeconds seconds: Double = 3.0) -> Window {
        let clamped = min(max(seconds, 2.0), 4.0)
        let frames = H3Const.alignFrameCount(UInt32(Double(H3Const.fps) * clamped))
        let latentT = H3Const.videoLatentT(frameCount: frames)
        let audioT = H3Const.audioLatentT(frameCount: frames)
        return Window(seconds: clamped, frames: frames, latentT: latentT, audioT: audioT)
    }

    // MARK: - 落盘（原子写：临时文件 + rename）

    /// 保存尾段 latent 为 .h3cc。
    /// - Parameter tailLatent: 已裁好的尾段 clean latent，形状 [winLatentT, latH, latW, latC]
    ///   （任意 dtype，内部统一转 fp16 存储）。
    /// - Parameter condRows: v2+ 本节点干净条件行 [rows, vpatch]（fp16 存储），
    ///   供下游节点直接拼接注入；nil = 不落盘条件行。
    /// - Parameter condIDs / condRowSpans: v2+ 条件资源去重元数据。
    ///   condRows 按资源顺序拼接，condIDs[i] 为该资源唯一 id（路径），
    ///   condRowSpans[i] 为对应行段行数；二者长度一致且 spans 之和 == rows。
    ///   无 condRows 时须传空数组。
    @discardableResult
    public static func save(nodeID: UUID, branchID: String?, rootDir: String,
                            tailLatent: MLXArray,
                            win: Window,
                            fingerprint: Expect,
                            latH: Int, latW: Int, latC: Int,
                            gridH: Int, gridW: Int,
                            videoPatchDim: Int,
                            condRows: MLXArray? = nil,
                            condIDs: [String] = [],
                            condRowSpans: [Int] = [],
                            createdAt: TimeInterval = Date().timeIntervalSince1970) throws -> URL {
        let expectedT = Int(win.latentT)
        guard tailLatent.ndim == 4,
              tailLatent.shape[0] == expectedT,
              tailLatent.shape[1] == latH,
              tailLatent.shape[2] == latW,
              tailLatent.shape[3] == latC else {
            throw NSError(domain: "H3ContinuationCache", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "save 形状不匹配：期望 [\(expectedT),\(latH),\(latW),\(latC)]，实际 \(tailLatent.shape)"])
        }

        // fp16 序列化（尾段 latent）
        let f16 = tailLatent.asType(.float16)
        MLX.eval(f16)
        let raw = f16.asArray(Float16.self)
        var latentData = Data(count: raw.count * 2)
        latentData.withUnsafeMutableBytes { dst in
            raw.withUnsafeBytes { src in
                dst.copyMemory(from: src)
            }
        }

        // v2+ 条件行 fp16 序列化（[rows, vpatch]）
        var condData: Data? = nil
        var condRowsCount: Int? = nil
        var condVPatch: Int? = nil
        if let condRows, condRows.ndim == 2, condRows.shape[0] > 0 {
            // 去重元数据一致性：ids 与 spans 等长、spans 之和 == rows
            if !condIDs.isEmpty || !condRowSpans.isEmpty {
                guard condIDs.count == condRowSpans.count,
                      condRowSpans.reduce(0, +) == condRows.shape[0] else {
                    throw NSError(domain: "H3ContinuationCache", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey:
                                    "save 去重元数据不一致：condIDs=\(condIDs.count) spans=\(condRowSpans) rows=\(condRows.shape[0])"])
                }
            }
            let cf16 = condRows.asType(.float16)
            MLX.eval(cf16)
            let cRaw = cf16.asArray(Float16.self)
            var cd = Data(count: cRaw.count * 2)
            cd.withUnsafeMutableBytes { dst in
                cRaw.withUnsafeBytes { src in
                    dst.copyMemory(from: src)
                }
            }
            condData = cd
            condRowsCount = condRows.shape[0]
            condVPatch = condRows.shape[1]
        }

        let header = Header(nodeID: nodeID, branchID: branchID,
                            winSeconds: win.seconds, winFrames: win.frames,
                            winLatentT: win.latentT, winLatentRows: win.rows(gridH: gridH, gridW: gridW),
                            winAudioT: win.audioT,
                            modelKey: fingerprint.modelKey, width: fingerprint.width,
                            height: fingerprint.height, steps: fingerprint.steps,
                            latentT: fingerprint.latentT, frameCount: fingerprint.frameCount,
                            refCount: fingerprint.refCount,
                            latH: latH, latW: latW, latC: latC,
                            gridH: gridH, gridW: gridW, videoPatchDim: videoPatchDim,
                            createdAt: createdAt,
                            condRowsRows: condRowsCount, condRowsVPatch: condVPatch,
                            condIDs: condIDs, condRowSpans: condRowSpans)

        let jsonData = try JSONEncoder().encode(header)
        var payload = Data()
        payload.append(contentsOf: magic)
        var headerLen = UInt64(jsonData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { payload.append(contentsOf: $0) }
        payload.append(jsonData)
        payload.append(latentData)
        if let condData { payload.append(condData) }

        let fm = FileManager.default
        let url = cacheURL(nodeID: nodeID, branchID: branchID, rootDir: rootDir)
        let dir = url.deletingLastPathComponent()
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let tmp = dir.appendingPathComponent(".tmp-" + UUID().uuidString)
        // 原子写：先写临时文件（.atomic 自带临时文件 + rename），再覆盖目标
        try payload.write(to: tmp, options: .atomic)
        if fm.fileExists(atPath: url.path) {
            try? fm.removeItem(at: url)
        }
        try fm.moveItem(at: tmp, to: url)
        return url
    }

    // MARK: - 读取（严格 fingerprint 校验）

    /// 读取 .h3cc。文件缺失 / 损坏 / 指纹或几何任一不匹配 → nil（调用方回退从头生成）。
    public static func load(nodeID: UUID, branchID: String?, rootDir: String,
                            expect: Expect,
                            shapeCheck: (latH: Int, latW: Int, latC: Int, gridH: Int, gridW: Int, videoPatchDim: Int)? = nil,
                            recovery: Recovery = Recovery()) -> Loaded? {
        let url = cacheURL(nodeID: nodeID, branchID: branchID, rootDir: rootDir)
        guard let payload = try? Data(contentsOf: url), payload.count >= magicByteCount + headerLenByteCount else {
            return nil
        }
        // magic
        let magicBytes = payload.prefix(magicByteCount)
        guard magicBytes.elementsEqual(magic) else { return nil }
        // headerLen
        var headerLenRaw = UInt64(0)
        let lenSlice = payload.subdata(in: magicByteCount..<(magicByteCount + headerLenByteCount))
        _ = withUnsafeMutableBytes(of: &headerLenRaw) { lenSlice.copyBytes(to: $0) }
        let headerLen = Int(UInt64(littleEndian: headerLenRaw))
        let headerStart = magicByteCount + headerLenByteCount
        guard headerLen > 0, headerStart + headerLen <= payload.count else { return nil }
        // header
        let jsonData = payload.subdata(in: headerStart..<(headerStart + headerLen))
        guard let header = try? JSONDecoder().decode(Header.self, from: jsonData),
              header.version == 1 || header.version == 2 else { return nil }
        // fingerprint
        guard header.modelKey == expect.modelKey,
              header.width == expect.width,
              header.height == expect.height,
              header.steps == expect.steps,
              header.latentT == expect.latentT,
              header.frameCount == expect.frameCount,
              header.refCount == expect.refCount else { return nil }
        // 几何
        if let sc = shapeCheck {
            guard header.latH == sc.latH, header.latW == sc.latW, header.latC == sc.latC,
                  header.gridH == sc.gridH, header.gridW == sc.gridW,
                  header.videoPatchDim == sc.videoPatchDim else { return nil }
        }
        // payload：latent 段 + （v2）条件段
        let latentData = payload.subdata(in: (headerStart + headerLen)..<payload.count)
        let expectedBytes = Int(header.winLatentT) * header.latH * header.latW * header.latC * 2
        guard latentData.count >= expectedBytes else { return nil }
        var condData: Data? = nil
        if header.version >= 2, let rows = header.condRowsRows, let vp = header.condRowsVPatch, rows > 0 {
            let condBytes = rows * vp * 2
            guard latentData.count >= expectedBytes + condBytes else { return nil }
            condData = latentData.subdata(in: expectedBytes..<(expectedBytes + condBytes))
        }
        let latentSlice = latentData.subdata(in: 0..<expectedBytes)
        guard latentSlice.count == expectedBytes else { return nil }
        return Loaded(header: header, latentData: latentSlice, recovery: recovery, condRowsData: condData)
    }

    // MARK: - 删除（幂等）

    /// 删除指定节点的 .h3cc。文件不存在时静默成功（幂等）。
    public static func delete(nodeID: UUID, branchID: String?, rootDir: String) {
        let url = cacheURL(nodeID: nodeID, branchID: branchID, rootDir: rootDir)
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - fp16 载荷重建

    /// 从 fp16 raw data 重建 MLXArray（[T,H,W,C]）。
    public static func mlxArray(fromData data: Data, shape: [Int]) -> MLXArray? {
        let expectedBytes = shape.reduce(1, *) * 2
        guard data.count == expectedBytes else { return nil }
        return MLXArray(data, shape, dtype: .float16)
    }
}
