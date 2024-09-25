import subprocess
from pathlib import Path
from importlib import metadata
import importlib.util


def install_packages(script_path):
    required = set()
    installed = {dist.metadata['Name'] for dist in metadata.distributions()}
    with open(script_path, 'r') as file:
        for line in file:
            if line.startswith('import ') or line.startswith('from '):
                package = line.split()[1].split('.')[0]
                # Check if package seems like a standard library module
                if package not in installed and not package.startswith(('sys', 'os', 'datetime')):
                    required.add(package)

    for package in required:
        if package == "PIL":
            continue
        if importlib.util.find_spec(package) is None:
            continue

        subprocess.run(["pip", "install", package], check=True)


def scan_directory(directory_path):
    pathlist = Path(directory_path).rglob('*.py')
    for path in pathlist:
        path_in_str = str(path)
        print(f"Installing packages for: {path_in_str}")
        install_packages(path_in_str)


# Replace 'path_to_directory' with the path to your directory
scan_directory('/home/bobsun/bob/convformer')

{"head": "X:0\nT:晴天\nQ:1/4=80\nM:4/4\nL:1/16\nK:C", "line": [{
                                                                   "melody": "|\"F9\"A2c2 g2c2 F2G1A1 g2c2 |\"C\"C2G2 g2c2 C2g2 B,2g2 |\"F9\"A2c2 g2c2 F2G1A1 g2c2 |\"C\"C2G2 g2c2 C2g2 B,2g2 |\"Am\"A,2c2 e2c2 \"F\"F2c2 f2c2 |\"C\"C2c2 e2c2 \"G\"G2d2 g2d2 |",
                                                                   "lyric": "w: "}, {
                                                                   "melody": "|\"Am\"A,2c2 e2c2 \"F\"F2c2 f2c2 |\"C\"C2c2 e2c2 \"G\"G2d2 g2d2 |\"Am\"z2G2 G1C1C2 \"F\"C4 D2E2 |\"C\"z2G2 G2C2 \"G\"C2D1E1 D1C1G,2 |\"Am\"z2G2 G2C2 \"F\"C4 D2E2 |\"C\"z2E4 D1E1\"G\"F1E1 D1F1E1D1 C2|",
                                                                   "lyric": "w: * * * * * * * * * * * * * * * * * 故 事 * 的 小 黄 花， * 从 出 生 那 年 就 飘 * 着。 * 童 年 的 荡 秋 千， * 随 记 忆 一 直 晃 到 现 * 在"},
                                                               {
                                                                   "melody": "|\"Am\"G,2C2 C2E2 \"F\"F2E2 D2C1D1 |\"C\"E2E2 E2E2 \"G\"D1E1D1C1 C4 |\"Am\"G,2C2 C2E2 \"F\"F2E2 D2C1D1 |\"C\"E2E2 E2E2 \"G\"D1E1D1C1 C2B,1|\"Am\"B,1C1C1C1 B,1C2C1 \"F\"C1C1C1C1 B,1C2C1 |",
                                                                   "lyric": "w: Re So So Si Do Si La, So La Si Si Si Si La Si La * So. 吹 着 前 奏 望 着 天 空， 我 想 起 花 瓣 试 着 掉 * 落。 为 你 翘 课 的 那 一 天， * 花 落 的 那 一 天"},
                                                               {
                                                                   "melody": "|\"C\"C1C1C1C1 B,1C2C1 \"G\"C1C1C1C1 G1G2G1 |\"Am\"G1G1G1G1 G1G2G1 \"F\"G1G1G1G1 G1F1F1E1 |\"C\"E4 \"G\"z4 C1C1C1C1 |\"Am\"A2B2 C2G2 \"F\"F2E2 C2C2 |\"C\"C4 z2B,2 \"G\"C1C1C1C1 E2C2 |",
                                                                   "lyric": "w: * 教 室 的 那 一 间， * 我 怎 么 看 不 见。 * 消 失 的 下 雨 天， * 我 好 想 再 淋 一 遍。 * * 没 想 到 失 去 的 勇 气 我 还 留 着， * * 啊 好 想 再 问 一 遍"},
                                                               {
                                                                   "melody": "|\"Am\"A,2B,2 C2G2 \"F\"F2E1C1 C2D2 |\"Gsus4\"D4 \"G\"z4 z4 |\"C\"E2D2 F2E4 C2G2 B2|\"Am\"c2B2 G2C4 C2A2 A2|\"F\"A2A2 G2\"G\"G4 G2F2 E2|\"C\"D2E2 F2E2 E4 |",
                                                                   "lyric": "w: 你 会 等 待 还 是 离 开。 * * * * 刮 风 这 天 我 试 过 握 着 你 手， 但 偏 偏 * 雨 渐 渐， 大 到 我 看 你 不 见。"},
                                                               {
                                                                   "melody": "|\"E7\"E2_G2 _A2E4 F2G2 B2|\"Am\"d2B2 c2c2 \"G\"c4 c2|\"F\"c2G2 G2A2 G1F1F2 D2E2 |\"G\"F2G2 A2C2 A2B1B4 |\"C\"E2D2 F2E2 z2C2 G2B2 |\"Am\"c2B2 G1C1C2 C2C2 A2A2 |",
                                                                   "lyric": "w: 还 要 多 久 我 才 能 在 你 身 边， * 等 到 放 晴 的 那 * 天 也 许 我 会 比 较 好 一 点。 从 前 从 前 * 有 个 人 爱 你 很 * 久， * 但 偏 偏"},
                                                               {
                                                                   "melody": "|\"F\"z2A2 G2G2 \"G\"z2G2 F2E2 |\"C\"D2E2 F2E2 E4 |\"E7\"E2_G2 _A2E2 E2F2 G2B2 |\"Am\"d2B2 c2c2 \"G\"c4 c2|\"F\"c2G2 G2A2 G1F1F2 A,2B,2 |\"G\"C2D2 E2D2 z2C2 E2C2 |",
                                                                   "lyric": "w: * 风 渐 渐， * 把 距 离 吹 得 好 远。 * 好 不 容 易 * 又 能 再 多 爱 一 天， * 但 故 事 的 最 后 * 你 好 像 还 是 说 了 * oh 拜 拜"},
                                                               {
                                                                   "melody": "|\"F9\"A2c2 g2c2 F2G1A1 g2c2 |\"C\"C2G2 g2c2 C2g2 B,2g2 |\"F9\"A2c2 g2c2 F2G1A1 g2c2 |\"C\"e4 |:",
                                                                   "lyric": "w: "}]},\
{"head": "X:1\nT:枫\nQ:1/4=80\nM:4/4\nL:1/16\nK:C", "line": [{
                                                                    "melody": "|\"Am\"c'2e1b1 b1c'2\"G\"b2 d1g1g4 |\"F\"a2c1g1 g2f2 \"Csus4\"f2G1e1 \"G/B\"e2d1e1 |\"Am\"c2d1e1 e2c'2 \"Cmaj7/E\"b2c'2 c'2b1c'1 |\"F\"A2C2 G1A1c2 |\"G13\"A4 B4 |:\"C9\"z2C2 C2C2 C2G,2 C2D2 |\"G/B\"D2D2 D2E2 D2G,2 G,4 |",
                                                                    "lyric": "w: * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 乌 云 在 我 们 心 里 搁 下 一 块 阴 影，"},
                                                                {
                                                                    "melody": "|\"Am\"c'2e1b1 b1c'2\"G\"b2 d1g1g4 |\"F\"a2c1g1 g2f2 \"Csus4\"f2G1e1 \"G/B\"e2d1e1 |\"Am\"c2d1e1 e2c'2 \"Cmaj7/E\"b2c'2 c'2b1c'1 |\"F\"A2C2 G1A1c2 |\"G13\"A4 B4 |:\"C9\"z2C2 C2C2 C2G,2 C2D2 |\"G/B\"D2D2 D2E2 D2G,2 G,4 |",
                                                                    "lyric": "w: * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 被 伤 透 的 心 能 不 能 够 继 续 爱 我，"},
                                                                {
                                                                    "melody": "|\"Am\"z2C2 C2C2 C2D2 E2F2 |\"C6/E\"F4 F2E1D1 \"Em7\"E4 z2E2 |\"F\"D2C1C1 C4 z4 z2E2 |\"C6/E\"D2C2 D2E1D1 \"Am\"C2C2 C4 |\"Dm7\"C2A,1C1 C2A,1C1 C2A,1C1 C2D1D1 |\"Gsus4\"D4 \"G\"z4 z4 :|",
                                                                    "lyric": "w: * 我 聆 听 沉 寂 已 久 的 * 心 * 情。 * 清 晰 透 明 * * * 就 像 美 丽 的 风 * 景， * 总 在 回 * 忆 里 * 才 看 * 的 清。"},
                                                                {
                                                                    "melody": "|\"Am\"z2C2 C2C2 C2D2 E2F2 |\"C6/E\"F4 F2E1D1 \"Em7\"E4 z2E2 |\"F\"D2C1C1 C4 z4 z2E2 |\"C6/E\"D2C2 D2E1D1 \"Am\"C2C2 C4 |\"Dm7\"C2A,1C1 C2A,1C1 C2A,1C1 C2D1D1 |\"Gsus4\"D4 \"G\"z4 z4 :|",
                                                                    "lyric": "w: * 我 用 力 牵 起 没 温 度 * 的 双 手。 * 过 往 温 柔 * * * 已 经 被 时 间 上 * 锁，"},
                                                                {
                                                                    "melody": "|\"Dm7\"C2A,1C1 C2A,1C1 \"F\"C2A,1C1 C1A2|\"Gsus4\"A2G2 G4 \"G\":\"C\"z2G,2 C2B,2 C2D2 E2C2 |\"G\"D2G2 G2G2 G2D2 C2B,2 |\"Am\"C2C2 C2B,2 C2D2 E2F2 |\"Em\"E2E2 E4 z2E2 E2G2 |",
                                                                    "lyric": "w: 只 剩 挥 * 散 不 * 去 的 * 难 过。 * * * 缓 缓 飘 落 的 枫 叶 像 思 * 念， * 我 点 燃 烛 火 温 暖 岁 末 的 秋 * 天。 * * 极 光 掠"},
                                                                {
                                                                    "melody": "|\"Dm7\"C2A,1C1 C2A,1C1 \"F\"C2A,1C1 C1A2|\"Gsus4\"A2G2 G4 \"G\":\"C\"z2G,2 C2B,2 C2D2 E2C2 |\"G\"D2G2 G2G2 G2D2 C2B,2 |\"Am\"C2C2 C2B,2 C2D2 E2F2 |\"Em\"E2E2 E4 z2E2 E2G2 |",
                                                                    "lyric": "w: * * * * * * * * * * * * * * * 缓 缓 飘 落 的 枫 叶 像 思 * 念， * 为 何 挽 回 要 赶 在 冬 天 来 之 * 前。 * * 爱 你 穿"},
                                                                {
                                                                    "melody": "|\"F\"G2A2 A2C2 C2A,2 C2A2 |\"Em\"A2G2 G2F1F1 \"Am\"F1E2E2 F1G1|\"Dm7\"G2F2 F2E1E1 E1D2z2 D1E1|\"G\"G2F1F1 F1E2E2 D1D1C2 C2:|\"G\"G2F1F1 F1E2D2 C2D2 C1C1|\"Am\"C4 |",
                                                                    "lyric": "w: 夺 天 * 边 * 北 风 掠 过 想 你 的 容 * 颜， * 我 把 爱 烧 成 了 落 * 叶， * 却 换 不 回 熟 * 悉 的 那 张 * 脸。 的 只 是 * 你 在 我 身 * 边。"},
                                                                {
                                                                    "melody": "|\"F\"G2A2 A2C2 C2A,2 C2A2 |\"Em\"A2G2 G2F1F1 \"Am\"F1E2E2 F1G1|\"Dm7\"G2F2 F2E1E1 E1D2z2 D1E1|\"G\"G2F1F1 F1E2E2 D1D1C2 C2:|\"G\"G2F1F1 F1E2D2 C2D2 C1C1|\"Am\"C4 |",
                                                                    "lyric": "w: 越 时 * 间 * 两 行 来 自 秋 末 的 眼 * 泪， * 让 爱 渗 透 了 地 面， * * * 我 要"},
                                                                {
                                                                    "melody": "|\"F\"e1c'1b1e1 c'1b1e1c'1 b4 |\"Am\"A,1e1c'1b1 e1c'1b1e1 d'1c'1b1c'1 b1g1d1e1 |\"F\"c1d1g2 c'1d'1g'2 g'4 |\"C\"z2C2 C2C2 C2G,2 C2D2 |\"G/B\"D2D2 D2E2 D2G,2 G,4 |\"Am\"z2C2 C2C2 C2D2 E2F2 |",
                                                                    "lyric": "w: * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 被 伤 透 的 心 能 不 能 够 继 续 爱 我， * * 我 用 力 牵 起 没 温"},
                                                                {
                                                                    "melody": "|\"C6/E\"F4 F2E1D1 \"Em7\"E4 z2E2 |\"F\"D2C1C1 C4 z4 z2E2 |\"C6/E\"D2C2 D2E1D1 \"Am\"C2C2 C4 |\"Dm7\"C2A,1C1 C2A,1C1 \"F\"C2A,1C1 C1A2|\"Gsus4\"A2G2 G4 \"G\"\"bD\"G,4 G,2C2 C2G,2 C2D2 |\"bG\"E4 E2D1A,1 A,4 z2D2 |",
                                                                    "lyric": "w: 度 * 的 双 手。 * 过 往 温 柔 * * * 已 经 被 时 间 上 * 锁， * 只 剩 挥 * 散 不 * 去 的 * 难 过。 * * 在 * 山 腰 间 飘 逸 的 * 红 雨， * * 随"},
                                                                {
                                                                    "melody": "|\"bAsus2\"D2C1C1 C2B,1B,1 B,4 z2D2 |\"bG\"D2C2 C2D2 \"bAsus2\"C2B,2 B,4 |\"bD\"C4 C2G2 G2F1E1 E1F2|\"bG\"G4 G2F1C1 C4 z2D1E1 |\"bAsus2\"F2E1E1 E2D1D1 D4 z2D1E1 |\"bG\"F2E1F1 E1D2\"bAsus2\"D2 C1D1D1C2 |",
                                                                    "lyric": "w: 着 北 风 * 凋 零， * * 我 轻 轻 摇 曳 风 * 铃。 想 * 唤 醒 被 遗 * 弃 的 * 爱 情， * * 雪 花 已 铺 满 * 了 地， * * 生 怕 窗 外 枫 * 叶 已 结 成 * 冰"},
                                                                {
                                                                    "melody": "|:\"bD\"z2G,2 C2B,2 C2D2 E2C2 |\"bAsus2\"D2G2 G2G2 G2D2 C2B,2 |\"bBm\"C2C2 C2B,2 C2D2 E2F2 |\"Fm\"E2E2 E4 z2E2 E2G2 |\"bG\"G2A2 A2C2 C2A,2 C2A2 |\"Fm\"A2G2 G2F1F1 \"bBm\"F1E2E2 F1G1|",
                                                                    "lyric": "w: * 缓 缓 飘 落 的 枫 叶 像 思 * 念， * 我 点 燃 烛 火 温 暖 岁 末 的 秋 * 天。 * * 极 光 掠 夺 天 * 边 * 北 风 掠 过 想 你 的 容 * 颜， * 我 把"},
                                                                {
                                                                    "melody": "|:\"bD\"z2G,2 C2B,2 C2D2 E2C2 |\"bAsus2\"D2G2 G2G2 G2D2 C2B,2 |\"bBm\"C2C2 C2B,2 C2D2 E2F2 |\"Fm\"E2E2 E4 z2E2 E2G2 |\"bG\"G2A2 A2C2 C2A,2 C2A2 |\"Fm\"A2G2 G2F1F1 \"bBm\"F1E2E2 F1G1|",
                                                                    "lyric": "w: * 缓 缓 飘 落 的 枫 叶 像 思 * 念， * 为 何 挽 回 要 赶 在 冬 天 来 之 * 前。 * * 爱 你 穿 越 时 * 间， * 两 行 来"},
                                                                {
                                                                    "melody": "|\"bEm7\"G2F2 F2E1E1 E1D2z2 D1E1|\"bAsus2\"G2F1F1 F1E2E2 D1D1C2 C2:|\"Fm\"A2G2 G2E1B1 |\"bBm\"B1c2c2 d2z4 z4 |\"bBm\"z4 z4 z4 z2F1G1 |\"bEm7\"G2F2 F2E1E1 E1D2z2 D1E1|",
                                                                    "lyric": "w: 爱 烧 成 了 落 * 叶， * 却 换 不 回 熟 * 悉 的 那 张 * 脸。 自 秋 末 的 眼 * 泪。 * * * * * * * * 让 爱 渗 透 了 地 面， * * * 我 要"},
                                                                {
                                                                    "melody": "|\"bAsus2\"G2F1F1 F1E2\"bA7\"D2 C2D2 C1C1|\"bD\"C4 z4 |\"bD\"C4 |",
                                                                    "lyric": "w: 的 只 是 * 你 在 我 身 * 边。"}]}
