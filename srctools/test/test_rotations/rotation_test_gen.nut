// Generates test positions and angles to verify srctools.vec calculations.


function test() {
	local vec;
	local DELTA = 15;
	local ent = Entities.CreateByClassname("info_target");
	ent.__KeyValueFromString("targetname", "test_target")

	SendToConsole("con_logfile rotation_data");
	printl("------ START DATA ------");

	for(local pitch=0; pitch < 360; pitch += DELTA) {
		for(local yaw=0; yaw < 360; yaw += DELTA) {
			for (local roll = 0; roll < 360; roll += DELTA) {
				ent.SetAngles(pitch, yaw, roll);

				print("| " + pitch + " " + yaw + " " + roll + "    ");
				vec = ent.GetForwardVector();
				print(vec.x + " " + vec.y + " " + vec.z + "    ");
				vec = ent.GetLeftVector();
				print(vec.x + " " + vec.y + " " + vec.z + "    ");
				vec = ent.GetUpVector();
				printl(vec.x + " " + vec.y + " " + vec.z);
			}
		}
		EntFireByHandle(self, "RunScriptCode", "resume tester", 0.1, null, null);
		yield;
	}
	printl("-------- END DATA -------");
	EntFireByHandle(ent, "Kill", "", 0, null, null);
	EntFire("!player", "Ignite");
}

tester <- test();
EntFireByHandle(self, "RunScriptCode", "resume tester", 0.10, null, null);
