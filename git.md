根据你提供的 `git status`，当前在 `main` 分支，有大量已修改文件和 2 个新文件。以下是针对该仓库的具体命令：

### 1. 本地提交并推送到 GitHub

```bash
# 添加所有修改和新文件
git add .

# 提交（请根据实际改动内容修改 commit message）
git commit -m "feat: update engine precision, training scripts and validation docs"

# 推送到远程 main 分支
git push origin main
```

> 💡 **提示**：由于改动文件较多（36个修改 + 2个新增），建议提交前用 `git diff --stat` 快速确认变更范围，确保没有误包含敏感配置或临时调试代码。

---

### 2. 远程服务器拉取最新代码

在远程服务器项目目录下执行：

```bash
# 标准拉取（推荐）
git pull origin main
```

如果远程服务器上存在本地未提交的修改导致 pull 冲突，可以先暂存再拉取：

```bash
git stash
git pull origin main
git stash pop
```

如果远程服务器的本地历史与远程严重分歧、且确认可以丢弃本地改动，强制同步到最新版本：

```bash
git fetch origin
git reset --hard origin/main
```

⚠️ `reset --hard` 会**永久丢弃**所有本地未提交的修改，仅在确认不需要保留本地改动时使用。