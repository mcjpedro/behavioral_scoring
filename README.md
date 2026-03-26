# Behavioral Scoring 

This is version 0.0.0.0.super.beta

If you need any help or find any bugs, please feel free to reach out to me (jcarvalh@umn.edu).

Use Nuitka to compile it into a C++ executable using the command:

```
python -m nuitka --standalone --output-dir=behavior_annotation --windows-disable-console --enable-plugin=tk-inter --enable-plugin=numpy --enable-plugin=matplotlib --include-package=pandas --include-package=cv2 behavior_annotation.py
```

After building the distribution you can open the software by running the “Application” (.exe) file on Windows. It should not damage your PC (hopefully!).

It may take a few seconds to start.
Performance may be slower on less powerful computers.
You can create a shortcut to the application in your desktop.

Manual scoring is really fun—enjoy your time! :)

Joao Pedro Carvalho Moreira, 3/26/2026
K-M Lab, University of Minnesota
