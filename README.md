# GlitchPy
GlitchPy is a project developed by Gus Becker to procedurally edit photos.

## Project Structure

File structure of the project is as follows:

```
.
├── glitchpy  # Directory contrianing custom modules
│   ├── __init__.py
│   ├── glitch.py
│   └── view.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Change Log
### v0.0.2
2024/01/19
- Add `utils.get_palette`, `utils.get_palette_df`, and `view.palette`
- Add `utils.reduce_color_by_rounding`

2024/01/19
- Add `glitch.split_hues_by_percentile` and `glitch.split_classes`
- Add `view.get_colors_by_count`
- Add `utils.save_images`, `utils.threshold_multi_otsu`, and `view.isolate_classes`
- Add `view.histogram_split`

2024/01/10
- Add `glitch.dilate_with_probability`

2024/01/09
- Add `view.images` to plot multiple images together
- Rewrite `glitch.add_noise` to accept a list of images

2024/01/04
- Add `glitch.posterize_otsu`
- Add `glitch.add_noise`
- Add `glitch.game_of_life`

