# Lanternwake: the prompts

The prompts and the pixel-art cleanup script behind **Lanternwake**, a side-scrolling
builder inspired by Kingdom Two Crowns, built with Claude Code, ChatGPT for the art,
and Godot 4.7.

From the episode on **LoFi Code Cafe**: *I Built a Kingdom Two Crowns-Style Game with Claude Code* (link coming when the video is live).

The game itself isn't open source. This repo is the part you can reuse.

## What's here

| File | What it is |
|---|---|
| [`prompts.md`](prompts.md) | The 15 prompts that shaped the game, copied from the Claude Code session log, with their times |
| [`tools/pixelize.py`](tools/pixelize.py) | Turns AI-generated "pixel art" sheets into clean sprites: removes the magenta background, rebuilds a real pixel grid, snaps every colour to one shared palette, and splits the sheet into sprites and animation strips |
| [`art/chatgpt-image-prompts.json`](art/chatgpt-image-prompts.json) | The image prompts used with ChatGPT for the sprite sheets and backgrounds |

## The real numbers

| | |
|---|---|
| First prompt → latest commit | 38 h 27 min (Nov 1, 19:44 → Nov 3, 10:11, UTC+2) |
| Time Claude was actually working | about 8.8 h (from the session log, pauses over 20 min left out) |
| Commits | 17, every one co-authored with Claude |
| GDScript | 8,853 lines in 50 files |
| Bugs shown in the video | 4 |

It isn't finished, and it isn't on Steam.

## Using pixelize.py

```bash
pip install -r requirements.txt
python tools/pixelize.py sheet.png --cell 9 --palette palette.png --names keeper*8 --out sprites
```

Ask your image generator for pixel art on a flat `#FF00FF` magenta background. `--cell`
is how many source pixels make one real pixel. Run `python tools/pixelize.py --help` for every option.

## License

MIT. See [LICENSE](LICENSE).
