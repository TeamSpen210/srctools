rem Call dmxconvert to get various versions of this. Hardcoded to my Steam Library for simplicity.
SET STEAMLIB=SteamLibrary/SteamApps/common
"F:/%STEAMLIB%/Half-Life 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v2.dmx
"S:/%STEAMLIB%/Left 4 Dead 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v4.dmx
"F:/%STEAMLIB%/Portal 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v5.dmx
