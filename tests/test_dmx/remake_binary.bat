rem Call dmxconvert to get various versions of this. Hardcoded to my Steam Library for simplicity.
SET STEAMLIB=F:/SteamLibrary/SteamApps/common
"%STEAMLIB%/Half-Life 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v2.dmx
"%STEAMLIB%/Left 4 Dead 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v4.dmx
"%STEAMLIB%/Portal 2/bin/dmxconvert.exe" -i keyvalues2.dmx -oe binary -o binary_v5.dmx
