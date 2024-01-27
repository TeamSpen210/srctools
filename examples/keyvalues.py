from srctools import Keyvalues

# Reading and writing
with open('filename.vdf', 'r') as f1:
	kv = Keyvalues.parse(f)
	
with open('filename.vdf', 'w') as f2:
	for line in kv.export():
		f2.write(line)

kv = Keyvalues('Root', [
	Keyvalues('block', [
		Keyvalues('value', 'hello'),
		Keyvalues('value', '42'),
	])
])

# Keyvalues can either be a block, or a single leaf.
kv.real_name # 'Root'
kv.name # 'root', casefolded.
block = kv.find_key('Block')  # Find specified key, or raise an exception.
block.find_key('value').value  # Gives the *last* key if multiple.
block['value']  # Fetch a key and return the string value.
block['notpresent', 'null']  # With two values, the second is a default.
block.int('value', -1)  # Shortcut to fetch a value, then parse. Returns default on failure.
# Also kv.bool(), kv.float(), kv.vec()
'value' in block