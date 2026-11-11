# 在 CTFY 上评测 prtseek

平台入口：<https://ctfy.tsinghua.edu.cn/evals/new>

## 一句话

CTFY 要的不是一个网址，而是一个 **一次性容器**：平台起容器 → 给环境变量和
`/ctfy/work/` 简报 → 你的进程干活 → 写 `/ctfy/work/result.json` → 退出。
协议全文见 `https://ctfy.tsinghua.edu.cn/docs/standards/CTFY-0001`。

prtseek 原本是常驻 Web 服务，所以新增了一个无头入口 `cmd/prtseek-ctfy`。

## 本地运行（不需要 Docker）

```bash
go build -o prtseek-ctfy.exe ./cmd/prtseek-ctfy

# 1) 协议一致性自检（无网络、无模型）
CTFY_WORK_DIR=./work CTFY_EVAL_DRY_RUN=1 ./prtseek-ctfy.exe

# 2) 用真实模型跑一次本地模拟题（平台 API 缺省时答案落进 result.json）
CTFY_WORK_DIR=./work \
LLM_FLAVOR=anthropic LLM_MODEL=claude-sonnet-5-5 LLM_API_KEY=sk-... \
./prtseek-ctfy.exe -role solver -max-turns 20
```

`work/` 里需要 `task.json`（可手写，字段见协议 §2）和可选的 `README.md`。

## 环境变量对照

| 平台给 | prtseek-ctfy 怎么用 |
|---|---|
| `CTFY_WORK_DIR` | 工作目录；缺省 `/ctfy/work` |
| `CTFY_PLATFORM_URL` + `CTFY_TOKEN` | 提交答案、刷新被锁住的小问 |
| `LLM_FLAVOR` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL` | 映射到 provider；`LLM_API_KEY` 为空时**拒绝运行**（协议 §1.1） |
| `CTFY_EVAL_MAX_TURNS` / `task.json:max_turns` | 轮次预算；null = 不限 |
| `CTFY_EVAL_DRY_RUN=1` | 写 `{"schema_version":3,"stop_reason":"dry_run"}` 后退出 0 |

## Agent 拿到的工具

在 SDK 默认工具（Bash/Read/Write/Edit/LS/Glob/Grep）之外，加了 5 个：

| 工具 | 作用 |
|---|---|
| `submit_answer` | 提交答案到平台；无凭证时记入 `result.json.answers` |
| `verify_answer` | 用判分器校验候选但不计入提交次数 |
| `refresh_questions` | 重读题目，解锁被前置条件挡住的小问 |
| `list_questions` | 看哪些小问已提交/已答对，避免重复提交 |
| `note_writeup` | 追加 writeup 到 `result.json` |

## 构建镜像

### A. 本机有任何 Docker 兼容 CLI

```bash
docker build -f Dockerfile.ctfy -t registry.ctfy.io/<你的命名空间>/prtseek-ctfy:v1 .
docker login registry.ctfy.io -u ctfy -p <pf_ 开头的过程令牌>
docker push registry.ctfy.io/<你的命名空间>/prtseek-ctfy:v1
```

`<你的命名空间>` 就是向导里显示的那个 `registry.ctfy.io/d3c1847d-.../`。
**不要**用默认的 `ghcr.io/me/agent:1` 占位地址。

### B. 本机没有 Docker（当前这台机器）

用 GitHub Actions 构建推送（仓库已公开，workflow 已就位）：

1. 仓库 Settings → Secrets → Actions → 新建 `CTFY_REGISTRY_TOKEN`，值填 CTFY 的 `pf_` 令牌。
2. Actions → **CTFY agent image** → Run workflow，填 `tag=v1` 和
   `repository_prefix=registry.ctfy.io/d3c1847d-c08f-42d1-b045-60e411aa8f26`。
3. 拿到镜像地址后，在向导「镜像地址」里填它，点「注册并检查」。

### C. 在别的机器上构建

任何有 docker 的机器上跑 A 的命令即可，仓库 `git clone` 下来不需要额外文件。

## 提交流程

1. 向导第 1 步：填镜像地址 → **注册并检查**（平台会做 manifest 读取 + 一次 dry run）。
2. 第 2 步：选模型。选平台出资的模型，`LLM_*` 由平台注入；用自己 key 时平台不翻译 API，要保持
   `LLM_FLAVOR` 与所选模型的「Served as」一致。
3. 第 3 步：开始。跑完在「我的测评」看分数、退出码、`/ctfy/work` 归档。

## 已知边界

- **一次运行只跑一轮**（`instance`/`qa` 模式平台会在解锁新小问后重开容器；本实现每轮重读 task.json）。
- 没有内置 `delegate_agents` 并行派发——`-role command` 的提示词会提到它，但无头入口未接该工具；
  想并行请用 `-role solver` 或直接让模型顺序推进。
- 镜像不含 Playwright/浏览器（约 1 GiB），需要浏览器自动化时自行改 Dockerfile.ctfy。
