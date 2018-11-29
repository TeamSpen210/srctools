// Generates test positions and angles to verify srctools.vec calculations.

// Makes an entity for the test, deleting old ones if required.
function make_ent(classname, name, x, y, z) {
	local ent = Entities.FindByName(null, name);
	if (!ent) {
		ent = Entities.CreateByClassname(classname);
	}
	ent.__KeyValueFromString("targetname", name);
	ent.SetOrigin(Vector(x, y, z));
	ent.SetAngles(0, 0, 0);
	return ent
}

ent_world <- make_ent("info_target", "test_world", 0.01, 0, 0);
ent_local <- make_ent("info_target", "test_local", 0.01, 0, 0);
POS <- [
	make_ent("info_target", "test_posx",  64, 0, 0),
	make_ent("info_target", "test_negx", -64, 0, 0),
	make_ent("info_target", "test_posy", 0,  64, 0),
	make_ent("info_target", "test_negy", 0, -64, 0),
	make_ent("info_target", "test_posz", 0, 0,  64),
	make_ent("info_target", "test_negz", 0, 0, -64),
];

EntFireByHandle(ent_local, "SetParent", "test_world", 0.01, null, null);
foreach (ent in POS) {
    EntFireByHandle(ent, "SetParent", "test_world", 0.01, null, null);
}

function test() {
	local ent;
	local origin;
	local full_ang;
	// SendToConsole("clear");
	SendToConsole("con_logfile rotation_data");
	printl("------ START DATA ------");

	for(local pitch=0; pitch < 360; pitch += 45) {
		for(local yaw=0; yaw < 360; yaw += 45) {
			for(local roll = 0; roll < 360; roll += 45) {
				printl("world: " + pitch + " " + yaw + " " + roll);
				ent_world.SetAngles(pitch, yaw, roll);

				foreach (ent in POS) {
					origin = ent.GetOrigin();
					printl("offset: " + ent.GetName() + " " +  origin.x + " " + origin.y + " " + origin.z);
				}
				for(local l_pitch=0; l_pitch < 360; l_pitch += 45) {
					for(local l_yaw=0; l_yaw < 360; l_yaw += 45) {
						for(local l_roll = 0; l_roll < 360; l_roll += 45) {
							ent_local.SetAngles(l_pitch, l_yaw, l_roll);
							full_ang = ent_local.GetAngles();
							printl("local: " + full_ang.x + " " + full_ang.y + " " + full_ang.z);
						}
						EntFireByHandle(self, "RunScriptCode", "resume tester", 0.1, null, null);
						yield;
					}
				}
			}
		}
	}
}

tester = test();
EntFireByHandle(self, "RunScriptCode", "resume tester", 0.10, null, null);