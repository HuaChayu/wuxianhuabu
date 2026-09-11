// ============================================================
//  文件作用：CanvasView 扩展。右上角颜色/比例按钮与下拉菜单、左下角工具栏
//  （网格吸附/显示网格/撤销/重做/缩放适配/导出/缩放控制）、分享画布。
//  互动文件：引用 画布 状态.swift（CanvasStore）；被 画布 ui.swift 引用
//  （topRightView、bottomLeftView）。
// ============================================================

import SwiftUI

// ============================================================

// MARK: - 右上角 / 左下角工具栏 + 颜色 / 比例菜单 + 分享

extension CanvasView {
    // MARK: - 右上角视图
    
    var topRightView: some View {
        HStack(spacing: 8) {
            // 颜色按钮（色块显示当前节点主题色）
            Button(action: {
                debugLog("工具栏：打开颜色菜单")
                withAnimation {
                    closeAllPanels()
                    showColorMenu.toggle()
                }
            }) {
                HStack(spacing: 4) {
                    Circle()
                        .fill(store.nodeColor)
                        .frame(width: 12, height: 12)
                    Text("颜色")
                }
                .font(.subheadline)
                .foregroundColor(.primary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .overlay(alignment: .topTrailing) {
                // 颜色选择面板（从颜色按钮下方延伸）
                if showColorMenu {
                    colorMenu
                        .offset(y: 34)
                }
            }
            
            // 比例按钮
            Button(action: {
                debugLog("工具栏：打开比例菜单")
                withAnimation {
                    closeAllPanels()
                    showRatioMenu.toggle()
                }
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "crop")
                    Text(store.currentRatio.rawValue)
                }
                .font(.subheadline)
                .foregroundColor(.primary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .overlay(alignment: .topTrailing) {
                // 比例选择菜单（从比例按钮下方延伸）
                if showRatioMenu {
                    ratioMenu
                        .offset(y: 34)
                }
            }
            
            // 秒数按钮（全局视频时长：5s / 10s；视频节点私有时长 nil 时跟随）
            Button(action: {
                debugLog("工具栏：打开时长菜单")
                withAnimation {
                    closeAllPanels()
                    showDurationMenu.toggle()
                }
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                    Text(store.currentDuration.rawValue)
                }
                .font(.subheadline)
                .foregroundColor(.primary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .overlay(alignment: .topTrailing) {
                // 时长选择菜单（从秒数按钮下方延伸）
                if showDurationMenu {
                    durationMenu
                        .offset(y: 34)
                }
            }
            
            // 连续视频模式开关：开 = 视频任务完成后沿输出连线自动触发下游视频节点生成
            Button(action: {
                debugLog("工具栏：连续视频 \(store.continuousVideoMode ? "关" : "开")")
                withAnimation {
                    store.continuousVideoMode.toggle()
                    store.save()
                }
            }) {
                HStack(spacing: 4) {
                    Text("连续视频")
                    Text(store.continuousVideoMode ? "开" : "关")
                        .foregroundColor(store.continuousVideoMode ? .blue : .secondary)
                }
                .font(.subheadline)
                .foregroundColor(.primary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(12)
    }
    
    // MARK: - 左下角视图（白色浮动工具栏，系统内置图标）
    
    var bottomLeftView: some View {
        HStack(spacing: 1) {
            // 网格吸附开关
            Button(action: {
                debugLog("工具栏：网格吸附 \(store.snapToGrid ? "关" : "开")")
                withAnimation { store.snapToGrid.toggle() }
                store.save()
            }) {
                Image(systemName: "square.grid.2x2")
                    .font(.system(size: 11))
                    .foregroundColor(store.snapToGrid ? .blue : .primary)
                    .frame(width: 24, height: 24)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("网格吸附")
            
            // 显示 / 隐藏网格
            iconToolbarButton(store.showGrid ? "eye" : "eye.slash",
                              color: store.showGrid ? .blue : .primary,
                              help: "显示网格") {
                debugLog("工具栏：显示网格 \(store.showGrid ? "关" : "开")")
                withAnimation { store.showGrid.toggle() }
                store.save()
            }
            
            Divider().frame(height: 12)
            
            // 撤销
            iconToolbarButton("arrow.uturn.backward",
                              color: store.canUndo ? .primary : .secondary,
                              help: "撤销",
                              disabled: !store.canUndo) {
                debugLog("工具栏：撤销")
                store.undo()
            }
            
            // 重做
            iconToolbarButton("arrow.uturn.forward",
                              color: store.canRedo ? .primary : .secondary,
                              help: "重做",
                              disabled: !store.canRedo) {
                debugLog("工具栏：重做")
                store.redo()
            }
            
            Divider().frame(height: 12)
            
            // 缩放适配（重置视图）
            iconToolbarButton("arrow.up.left.and.arrow.down.right",
                              help: "缩放适配") {
                debugLog("工具栏：缩放适配")
                withAnimation(canvasViewAnimation) {
                    store.reset()
                    store.save()
                }
            }
            
            // 导出
            iconToolbarButton("square.and.arrow.down",
                              help: "导出") {
                debugLog("工具栏：导出画布")
                shareCanvas()
            }
            
            Divider().frame(height: 12)
            
            // 缩放控制
            HStack(spacing: 1) {
                Button(action: {
                    debugLog("工具栏：缩小")
                    withAnimation {
                        store.pushSnapshot()
                        store.zoom = max(0.1, store.zoom - 0.1)
                        store.save()
                    }
                }) {
                    Image(systemName: "minus")
                        .font(.system(size: 9, weight: .bold))
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                
                Text(String(format: "%.0f%%", store.zoom * 100))
                    .font(.system(size: 10))
                    .frame(minWidth: 30)
                
                Button(action: {
                    debugLog("工具栏：放大")
                    withAnimation {
                        store.pushSnapshot()
                        store.zoom = min(3.0, store.zoom + 0.1)
                        store.save()
                    }
                }) {
                    Image(systemName: "plus")
                        .font(.system(size: 9, weight: .bold))
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 3)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.white.opacity(0.95))
                .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
        .padding(8)
    }
    
    // MARK: - 颜色选择面板（从右上角颜色按钮下方延伸）
    
    var colorMenu: some View {
        VStack(spacing: 4) {
            ForEach(Array(nodeColors.enumerated()), id: \.offset) { _, color in
                Button(action: {
                    debugLog("工具栏：选择颜色 \(colorName(color))")
                    withAnimation {
                        store.nodeColor = color // 计算属性 setter → AppSettings.setNodeColor，公共颜色源同步
                        store.save()
                    }
                    showColorMenu = false
                }) {
                    HStack(spacing: 12) {
                        // 颜色圆点
                        Circle()
                            .fill(color)
                            .frame(width: 18, height: 18)
                            .overlay(
                                Circle().stroke(Color.black.opacity(0.15), lineWidth: 1)
                            )
                        
                        Text(colorName(color))
                            .font(.subheadline)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        if store.nodeColor == color {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(.black)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(
                        hoveredColor == color ? Color.black.opacity(0.1) :
                        (store.nodeColor == color ? Color.black.opacity(0.06) : Color.clear)
                    )
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .onHover { hovering in
                    hoveredColor = hovering ? color : nil
                }
            }
        }
        .padding(10)
        .background(Color.white)
        .cornerRadius(12)
        .shadow(radius: 12)
        .frame(width: 150)
    }
    
    // 可选节点主题色
    var nodeColors: [Color] {
        [.pink, Color(red: 0.98, green: 0.65, blue: 0.72), .blue, .red, .orange, .green, .purple, .teal, .black]
    }
    
    // 颜色显示名
    func colorName(_ color: Color) -> String {
        if color == Color(red: 0.98, green: 0.65, blue: 0.72) { return "标准粉" }
        switch color {
        case .blue: return "蓝色"
        case .red: return "红色"
        case .orange: return "橙色"
        case .green: return "绿色"
        case .purple: return "紫色"
        case .pink: return "粉色"
        case .teal: return "青色"
        case .black: return "黑色"
        default: return "自定义"
        }
    }
    
    // MARK: - 比例选择菜单（从右上角按钮下方延伸）
    
    var ratioMenu: some View {
        VStack(spacing: 4) {
            ForEach(CanvasStore.Ratio.allCases, id: \.self) { ratio in
                Button(action: {
                    debugLog("工具栏：选择比例 \(ratio.displayName)")
                    withAnimation {
                        store.pushSnapshot()
                        store.currentRatio = ratio
                        // 遍历场景：所有已设私有比例的视频节点统一改为所选比例
                        for i in store.nodes.indices where store.nodes[i].ratio != nil {
                            store.nodes[i].ratio = ratio
                        }
                        store.save()
                    }
                    showRatioMenu = false
                }) {
                    HStack(spacing: 12) {
                        // 比例矩形图标
                        RoundedRectangle(cornerRadius: 2)
                            .stroke(Color.primary, lineWidth: 1.5)
                            .frame(width: ratioIconSize(ratio).width,
                                   height: ratioIconSize(ratio).height)
                        
                        Text(ratio.displayName)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        if store.currentRatio == ratio {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(.black)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(
                        hoveredRatio == ratio ? Color.black.opacity(0.1) :
                        (store.currentRatio == ratio ? Color.black.opacity(0.06) : Color.clear)
                    )
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .onHover { hovering in
                    hoveredRatio = hovering ? ratio : nil
                }
            }
        }
        .padding(10)
        .background(Color.white)
        .cornerRadius(12)
        .shadow(radius: 12)
        .frame(width: 210)
    }
    
    // MARK: - 时长选择菜单（全局视频秒数：5s / 10s；逻辑同比例：选全局时同步所有已设私有时长的节点）
    
    var durationMenu: some View {
        VStack(spacing: 4) {
            ForEach(VideoDuration.allCases, id: \.self) { d in
                Button(action: {
                    debugLog("工具栏：选择时长 \(d.displayName)")
                    withAnimation {
                        store.pushSnapshot()
                        store.currentDuration = d
                        // 遍历场景：所有已设私有时长的视频节点统一改为所选时长
                        for i in store.nodes.indices where store.nodes[i].duration != nil {
                            store.nodes[i].duration = d
                        }
                        store.save()
                    }
                    showDurationMenu = false
                }) {
                    HStack(spacing: 12) {
                        Text(d.displayName)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        if store.currentDuration == d {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(.black)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(
                        hoveredDuration == d ? Color.black.opacity(0.1) :
                        (store.currentDuration == d ? Color.black.opacity(0.06) : Color.clear)
                    )
                    .cornerRadius(6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .onHover { hovering in
                    hoveredDuration = hovering ? d : nil
                }
            }
        }
        .padding(10)
        .background(Color.white)
        .cornerRadius(12)
        .shadow(radius: 12)
        .frame(width: 140)
    }
    
    // 比例图标尺寸（保持比例，最长边 24）
    func ratioIconSize(_ ratio: CanvasStore.Ratio) -> CGSize {
        let maxSide: Double = 24
        let w = ratio.width
        let h = ratio.height
        let scale = maxSide / max(w, h)
        return CGSize(width: w * scale, height: h * scale)
    }
    
    // 分享画布
    func shareCanvas() {
        let text = "我的画布：\(canvasName)"
        let picker = NSSharingServicePicker(items: [text])
        if let window = NSApp.keyWindow, let contentView = window.contentView {
            picker.show(relativeTo: .zero, of: contentView, preferredEdge: .minY)
        }
    }
    
    // MARK: - 左下角 24×24 图标工具栏按钮统一封装
    
    func iconToolbarButton(
        _ systemName: String,
        color: Color = .primary,
        help: String? = nil,
        disabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11))
                .foregroundColor(color)
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .help(help ?? "")
    }
}

// MARK: - 比例菜单显示名（图2样式）

extension CanvasStore.Ratio {
    var displayName: String {
        switch self {
        case .ratio9_16: return "9:16（竖屏）"
        case .ratio16_9: return "16:9（横屏）"
        case .ratio21_9: return "21:9（电影）"
        case .ratio3_4: return "3:4"
        case .ratio4_3: return "4:3"
        case .ratio1_1: return "1:1"
        }
    }
}

