# 实验七第二部分：RAG 系统构建结果

生成时间：2026-06-04 11:14:36
LLM 后端：deepseek / deepseek-v4-flash
Embedding 后端：mock / mock-hash-embedding
向量数据库：numpy fallback

## 知识库文档列表及内容摘要

| 文档 | 摘要 |
|---|---|
| company_leave_policy.txt | 公司请假制度

第一条 适用范围：本制度适用于公司全体正式员工、试用期员工以及经部门负责人确认的实习人员。员工因病、因事、婚丧、生育、学习考试等原因不能按时出勤时，应按照本制度办理... |
| office_equipment_policy.txt | 办公设备申领与归还流程

第一条 设备范围：本流程所称办公设备包括笔记本电脑、台式机、显示器、键盘鼠标、耳机、移动硬盘、投影仪、会议摄像头以及经行政部登记的其他办公资产。设备由行政... |
| overtime_allowance_policy.txt | 公司加班与补贴政策

第一条 加班定义：加班是指员工因工作需要，在标准工作时间之外继续完成经批准的工作任务。员工自愿延长工作时间但未经过审批的，不计为公司认可的加班。所有加班应坚持... |
| remote_work_policy.txt | 远程办公与信息安全规定

第一条 申请条件：远程办公适用于因项目协作、出差、特殊天气、临时照护家庭成员或其他经公司认可的情形。员工申请远程办公时，应明确远程日期、工作地点、主要任务... |

## 分块统计

共加载 4 个文档，切分为 8 个文档块。

## 测试问题及模型回答记录表

| 类型 | 问题 | 模型回答 | 来源引用 |
|---|---|---|---|
| 直接问题 | 病假需要在什么时候提交申请，返岗后需要补交什么材料？ | 回答：病假需在上班前通过企业微信提交申请，返岗后应在三个工作日内补交医院诊断证明、病历或正规医疗机构开具的休息建议。<br><br>依据：company_leave_policy.txt | company_leave_policy.txt(company_leave_policy_chunk_001, score=0.4307)；company_leave_policy.txt(company_leave_policy_chunk_000, score=0.3905)；office_equipment_policy.txt(office_equipment_policy_chunk_000, score=0.3543)；overtime_allowance_policy.txt(overtime_allowance_policy_chunk_000, score=0.3445) |
| 直接问题 | 工作日晚间加班满两小时有什么补贴？超过四小时并晚于二十二点结束怎么处理？ | 回答：工作日晚间加班满两小时，给予三十元餐补。若超过四小时且结束时间晚于二十二点，可报销单程交通费。<br><br>依据：overtime_allowance_policy.txt | overtime_allowance_policy.txt(overtime_allowance_policy_chunk_000, score=0.5963)；remote_work_policy.txt(remote_work_policy_chunk_000, score=0.4563)；overtime_allowance_policy.txt(overtime_allowance_policy_chunk_001, score=0.4408)；company_leave_policy.txt(company_leave_policy_chunk_001, score=0.4239) |
| 综合问题 | 员工周末加班后想在下周调休，应该怎么申请，调休和加班记录之间有什么关系？ | 回答：员工周末加班后如需在下周调休，应在请假系统中选择“调休假”类型，并关联已审批的加班单来完成申请。调休必须基于已审批的加班记录，且周末加班形成的调休应在三个月内使用。<br><br>依据：overtime_allowance_policy.txt；company_leave_policy.txt | overtime_allowance_policy.txt(overtime_allowance_policy_chunk_000, score=0.5561)；overtime_allowance_policy.txt(overtime_allowance_policy_chunk_001, score=0.5327)；company_leave_policy.txt(company_leave_policy_chunk_001, score=0.4953)；remote_work_policy.txt(remote_work_policy_chunk_000, score=0.4191) |
| 综合问题 | 新员工需要领电脑并偶尔远程办公时，设备申领和信息安全方面分别要注意什么？ | 回答：新员工申领电脑时，须由直属主管提前三个工作日提交设备申领单，行政部按标准配置电脑和外设，信息技术部完成系统初始化、杀毒软件安装、磁盘加密和公司邮箱配置，员工领取时需核对资产编号并签署《办公资产领用确认单》。对于远程办公，原则上应使用公司发放的电脑，必须连接公司VPN或经批准的安全访问通道，不得通过公共网盘或个人邮箱等未授权方式传输公司文件；处理重要数据时要开启磁盘加密和屏幕锁定，离开座位时及时锁屏。<br><br>依据：[来源1] remote_work_policy.txt；[来源2] office_equipment_policy.txt | remote_work_policy.txt(remote_work_policy_chunk_000, score=0.5634)；office_equipment_policy.txt(office_equipment_policy_chunk_000, score=0.5338)；remote_work_policy.txt(remote_work_policy_chunk_001, score=0.4819)；overtime_allowance_policy.txt(overtime_allowance_policy_chunk_000, score=0.4346) |
| 知识库外 | 公司的股票期权什么时候兑现？ | 知识库中未找到与该问题直接相关的政策依据，因此无法回答。建议咨询人力资源部或查看补充制度文件。 | office_equipment_policy.txt(office_equipment_policy_chunk_000, score=0.3114)；remote_work_policy.txt(remote_work_policy_chunk_001, score=0.3113)；remote_work_policy.txt(remote_work_policy_chunk_000, score=0.3032)；overtime_allowance_policy.txt(overtime_allowance_policy_chunk_001, score=0.2982) |

## 简要分析

直接问题通常能命中单个制度片段，回答较稳定；综合问题需要把请假、加班、设备、远程办公等片段合并，来源引用能帮助核查；知识库外问题应拒答，避免模型凭常识编造公司制度。

## 核心流程说明

1. 读取 TXT/PDF 文档。
2. 按固定长度与重叠窗口切分文档块。
3. 通过 EmbeddingClient 生成向量。
4. 使用 FAISS IndexFlatIP 存储并按相似度检索。
5. 将检索片段作为上下文交给 LLM 生成答案并保留来源。