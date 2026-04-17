# AlphaFold2 Visually —— 交互式 3D 算法讲解项目

## 项目一句话

做一个类似 [bbycroft.net/llm](https://bbycroft.net/llm) 风格的交互式网页,用 3D 动画完整讲解 AlphaFold2 的算法流程(MSA → Evoformer → Structure Module → 蛋白结构输出),纯前端,可以滚动浏览、可以旋转缩放、可以单步播放每个模块的内部计算。

**定位**:这不是分子结构查看器(Mol*、PyMOL 已经做得很好),而是**算法本身的可视化**——把 Evoformer、triangle attention、IPA 这些抽象张量操作变成可以看见、可以转的 3D 场景。目前这个空缺没人填。

**参考物**(务必先看):
- https://bbycroft.net/llm —— 视觉风格和交互范式直接对标
- https://distill.pub —— 讲解叙事风格
- https://ciechanow.ski —— 滚动式 3D 讲解的最高水准

---

## 技术栈

- **框架**:React + Vite + TypeScript
- **3D 引擎**:Three.js + react-three-fiber + drei
- **动画**:framer-motion(UI) + gsap 或自写 tween(场景)
- **滚动驱动**:react-scroll-parallax 或 @react-three/drei 的 ScrollControls
- **状态**:zustand(轻量,适合 3D 场景)
- **部署**:GitHub Pages / Vercel 纯静态

**先不要**引入 Rust/Wasm、不要跑真实模型推理、不要后端。阶段 1 全部用预计算/合成数据驱动可视化。

---

## 阶段划分

### 阶段 1:单模块 MVP(目标 2-3 周)

**只做一个模块:Evoformer 的 Triangle Attention**。做到惊艳就停,不要贪多。

理由:Triangle Attention 是 AlphaFold 最 elegant 的设计,视觉上天然适合 3D(三个残基构成三角形,pair representation 是一个 2D 矩阵平面),讲清楚了观众有"原来如此"的顿悟感。

**交付物**:一个单页网页,一个小蛋白(20 个残基)的 triangle attention 过程动画。

### 阶段 2:完整流程串联(阶段 1 完成后再规划)

加入 MSA embedding、完整 Evoformer、Structure Module、Recycling。模块之间用相机过渡动画衔接。

### 阶段 3:真实数据 + 模型推理(可选,远期)

引入 Rust/Wasm 跑 ESMFold 或简化版推理,用户输入序列看真实过程。这一步再考虑性能问题。

**本次计划书只覆盖阶段 1。**

---

## 阶段 1 详细任务

### 项目初始化

- 用 Vite + React + TypeScript 模板起项目
- 安装依赖:`three`, `@react-three/fiber`, `@react-three/drei`, `zustand`, `leva`(调试面板)
- 配置 ESLint + Prettier
- 建立目录结构:
  ```
  src/
    scenes/          # 各个 3D 场景组件
    primitives/      # 可复用的 3D 原件(残基节点、连接线、热力图平面等)
    data/            # 合成数据、预计算张量
    store/           # zustand 状态
    ui/              # 2D UI 覆盖层(标题、说明文字、控制条)
    utils/           # 工具函数
  ```

### 核心可视化元素(primitives)

实现以下可复用 3D 组件,每个组件接受 props 控制外观,暴露动画接口:

1. **`<Residue>`**:单个残基节点,球体 + 标签(氨基酸字母),支持 highlight 状态(发光、缩放、颜色)
2. **`<ResidueChain>`**:一串残基排成链,间距可配,支持整条链的入场动画
3. **`<PairMatrix>`**:N×N 的 2D 热力图平面,悬浮在 3D 空间里,每个 cell 是一个小方块,颜色由数值映射(使用 diverging colormap,蓝-白-红)。支持高亮特定 (i, j) cell、支持整个矩阵的 shader 动画(波纹、脉冲)
4. **`<MSAStack>`**:N_seq × N_res 的堆叠条带(阶段 1 可以简化为一个纹理平面)
5. **`<TriangleHighlight>`**:在 pair matrix 上高亮三个点 (i,j), (j,k), (i,k) 形成的三角形,带发光边 + 脉冲动画
6. **`<AttentionFlow>`**:从源到目标的粒子流动画,表示信息传递

所有 primitive 用 react-three-fiber 写,避免命令式 Three.js 代码。

### Triangle Attention 场景设计

场景布局(从观众视角):

- **左侧**:一条水平排列的残基链 `<ResidueChain>`,20 个残基
- **中央**:悬浮的 `<PairMatrix>` 20×20 热力图平面,略微倾斜对着相机
- **右侧**:一个信息面板(HTML overlay),显示当前步骤的公式和文字说明

动画时间线(用户通过滚动或点击"下一步"触发):

1. **入场**:残基链从左侧淡入,pair matrix 从中央展开
2. **Step 1 - 选中更新目标**:高亮 pair matrix 上的某个 cell `(i, j)`,说"我们要更新这对残基的表示"
3. **Step 2 - 遍历第三个残基 k**:对每个 k,在残基链上高亮第三个残基,在 pair matrix 上画出 `(i,k)` 和 `(j,k)` 两个 cell,用 `<TriangleHighlight>` 显示三角形
4. **Step 3 - 三角形信息汇聚**:粒子从 `(i,k)` 和 `(j,k)` 流向 `(i,j)`,`(i,j)` cell 颜色渐变(表示被更新)
5. **Step 4 - 扫过所有 k**:快速播放 k 从 1 到 N 的过程,最后 `(i,j)` 定格到新值
6. **Step 5 - 对整个矩阵做**:整个 pair matrix 做一波"刷新"动画,所有 cell 按顺序更新

每一步在右侧面板显示对应的数学公式(用 KaTeX 渲染)和一句话解释。

### 交互控制

- **OrbitControls**:可以自由旋转、缩放、平移场景
- **播放控制条**:底部固定一个控制条,有播放/暂停/上一步/下一步/速度调节
- **自动演示模式**:默认自动播放一轮,用户可以随时介入暂停
- **调试面板**(开发用):leva,调节动画速度、颜色、相机参数

### 数据

阶段 1 全部用合成数据,不需要真实 AlphaFold 权重:

- 20 个残基,随机生成氨基酸序列
- pair representation 用合成矩阵(可以用简单的正弦 + 噪声生成)
- triangle attention 的更新值不需要真实计算,用视觉合理的数值驱动动画即可(重点是讲清楚"信息怎么流",不是数值准确)

在 `src/data/synthetic.ts` 里导出这些合成数据。

### 视觉风格规范

参考 bbycroft.net/llm 的基调:

- **背景**:深色(`#0a0a15` 或类似),带微弱的网格或星空
- **主色**:青色/品红 diverging colormap(用于 pair matrix)
- **强调色**:暖白光(用于 highlight、粒子流)
- **字体**:等宽字体(JetBrains Mono / Fira Code)配合 sans-serif(Inter)
- **后期处理**:bloom(发光)、景深、微弱色差。用 `@react-three/postprocessing`
- **镜头语言**:相机略微缓慢移动,让静态场景也有生命感(用 drei 的 `CameraShake` 或自写 lerp)

### UI 覆盖层

顶部:标题 "AlphaFold2 Visually — Triangle Attention"
右侧浮动面板:当前步骤文字 + 公式
底部:播放控制条
左上:回到顶部/跳转到其他模块(阶段 1 只有一个模块,但留接口)

---

## 具体执行顺序(给 Claude Code 的执行步骤)

1. 初始化 Vite + React + TS 项目,安装所有依赖,配置 tsconfig 和 eslint
2. 搭建基础 3D 场景:Canvas + 深色背景 + OrbitControls + 环境光 + 一个测试球体,确认渲染正常
3. 实现 `<Residue>` 和 `<ResidueChain>` primitive,放到场景里看效果
4. 实现 `<PairMatrix>` primitive,用 InstancedMesh 渲染 N×N cell,shader 控制颜色(diverging colormap)
5. 实现合成数据生成 `src/data/synthetic.ts`
6. 把残基链和 pair matrix 在场景里组合好,调相机和布局
7. 实现 zustand store,管理当前动画步骤(`currentStep`)和播放状态
8. 实现 `<TriangleHighlight>` 和 `<AttentionFlow>` primitive
9. 串联完整动画时间线,用 `useFrame` 或 gsap timeline 驱动,根据 `currentStep` 播放对应片段
10. 加入播放控制条 UI 和右侧说明面板(KaTeX 渲染公式)
11. 加入 postprocessing(bloom 等),调整视觉质感
12. 写 README,部署到 Vercel / GitHub Pages

**每完成一步跑一次 `npm run dev`,视觉效果不对不要继续往下走。**

---

## 成功标准

阶段 1 完成的标准:

1. 任何人打开网页,不用读文字,光看动画能感觉到"哦三个残基之间在传消息"
2. 读完右侧说明 + 看完动画,能理解 triangle attention 的直觉
3. 场景可以自由旋转,从各个角度看都不穿帮
4. 在 MacBook 集显上 60fps 流畅运行
5. 整个 bundle gzip 后 < 1MB(不算 Three.js)
6. 能发 Twitter / Hacker News 不丢人

---

## 不要做的事(阶段 1 明确排除)

- 不要做真实 AlphaFold 推理
- 不要引入 Rust / Wasm
- 不要做 MSA、Structure Module、Recycling(阶段 2 才做)
- 不要支持用户输入任意蛋白(阶段 3 才做)
- 不要做移动端适配(先把桌面端做好)
- 不要做国际化
- 不要写单元测试(这是视觉项目,靠眼睛验收)
- 不要过度抽象,代码能跑、结构清楚就行

---

## 给 Claude Code 的元指令

- 每次只做一小步,跑起来看效果,再做下一步
- 遇到视觉效果不对,截图贴出来讨论,不要闷头改
- Three.js / r3f 的 API 不确定时,查官方文档(https://threejs.org/docs、https://r3f.docs.pmnd.rs),不要猜
- 代码风格偏函数式,少用 class,多用 hooks
- 所有"魔数"(相机位置、动画时长、颜色值)放到 `src/config.ts` 里集中管理,方便调
- 写完一个场景后,录一小段 gif 给我看
