# surface waves
Code used to analyze surface waves from movies of colloidal fluids.
A new subdirectory will be created called "h_analysis" under the input directory.
Data will be stored here.

### Run ``analyze_boundary.py`` with path to directory holding frames of movie

`analyze_boundary.py /path/to/directory`

A new directory under the input directory will be created with the name "h_analysis". 
Data will be stored here

Optional parameters:

- `start` : First frame to analyze. Defaults to first frame. Accepts ```int```.

- `end`: Last frame to analyze. Defaults to last frame. Accepts ```int```.

- `extension`: File extension of images. Defaults to ``jpg``.

### To run with these parameters:
`analyze_boundary.py /path/to/directory --start startframe --end endframe --extension ext`

<img src="height.gif" width="200"/>   
