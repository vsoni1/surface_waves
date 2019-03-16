# surface waves

<img src="height.gif" width="500"/>   

Code used to analyze surface waves from movies of colloidal fluids.

### Run ``analyze_boundary.py`` with path to directory holding frames of movie

`analyze_boundary.py /path/to/directory`

Optional parameters:

- `start` : First frame to analyze. Defaults to first frame. Accepts ```int```.

- `end`: Last frame to analyze. Defaults to last frame. Accepts ```int```.

- `extension`: File extension of images. Defaults to ``jpg``.

### To run with these parameters:
`analyze_boundary.py /path/to/directory --start startframe --end endframe --extension ext`
