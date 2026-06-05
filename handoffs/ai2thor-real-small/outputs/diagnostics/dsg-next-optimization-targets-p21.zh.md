# P21 后 DSG 下一步优化目标

## 当前基线

- VLM-only p14 semantic match: 24/60 = 0.400000
- DSG p13 semantic match: 23/60 = 0.383333
- DSG - VLM delta: -1 case
- paired wins/losses/ties: 12 / 13 / 35

本报告不是最终研究结论。P21 的 VLM retry prediction 还没有返回，因此这里只整理下一轮 DSG 优化优先级。

## P0：目标和支撑物都存在，但缺 ON 边

这类 case 数：3。它们最适合作为下一轮 DSG 的第一批优化，因为不需要新增目标对象，只需要改进图构建中的支撑关系推断。

关键 case：

- `ai2thor-real-small-episode-002:FloorPlan201:0003:object_location:book_01_87_00_68_01_17:observation_aware:200007`: book ON chair
- `ai2thor-real-small-episode-003:FloorPlan301:0006:object_location:book_00_90_00_56_01_18:observation_aware:100007`: book ON bed
- `ai2thor-real-small-episode-004:FloorPlan401:0004:object_location:cloth_00_27_00_04_01_02:observation_aware:200010`: cloth ON bathtub

## P1：缺支撑面对象

这类 case 数：14。其中 `countertop` 占 9 条，是最值得优先提高 detector/RGB-D recall 的支撑面。

Top missing supports:

- countertop: 9
- dresser: 2
- shelf: 1
- coffeetable: 1
- handtowelholder: 1

## P2：缺目标对象

这类 case 数：15。这些 case 无法靠 GraphTool 后处理解决，必须让 observation-backed DSG 真正包含目标对象。

Top missing targets:

- creditcard: 3
- butterknife: 1
- bowl: 1
- cd: 1
- dishsponge: 1
- diningtable: 1
- keychain: 1
- laptop: 1
- mug: 1
- newspaper: 1
- papertowelroll: 1
- pen: 1
- pencil: 1

## 下一步

先等待 P21 ready35 的 35 条真实 VLM retry prediction 返回，并使用 scoped merge gate 重算 VLM-only。当前 scoped pending merge 已证明：如果真实返回缺失，VLM-only 仍是 24/60，不会被误报为提升；唯一 out-of-scope failed case 是缺 target crop 的 `cd`。若 VLM-only 上升，DSG 需要至少优先完成 P0 的 3 条 support-present relation missing case，再进入 P1/P2 的 detector coverage 工作。

JSON 报告：`dsg-next-optimization-targets-p21.json`
Report digest: `5fb131fca652aa14a45e7b4e49e761a85e614a21c71db3f7d0588287ff25fde5`
