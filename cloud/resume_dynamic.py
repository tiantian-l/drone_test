"""Resume a dynamic run without reconstructing or overwriting its settings."""
import sys
from pathlib import Path
import elements
from ruamel import yaml
from dreamerv3.main import main

logdir, steps, dry_run = sys.argv[1:]
config = elements.Config(yaml.YAML(typ="safe").load(
    (Path(logdir) / "config.yaml").read_text()))
if (not config.env.drone.dynamic_obstacles_enabled or
        config.env.drone.dynamic_obstacle_motion != "collision_reverse"):
    raise SystemExit("Refusing to resume a non-collision-reverse dynamic run")
config = config.update({"logdir": str(Path(logdir).resolve()),
                        "run.steps": int(steps), "run.from_checkpoint": ""})
argv = ["--configs", "defaults"]
def flatten(mapping, prefix=""):
    for key, value in mapping.items():
        key = prefix + key
        if hasattr(value, "items"):
            yield from flatten(value, key + ".")
        else:
            yield key, value

for key, value in flatten(config):
    if isinstance(value, (list, tuple)):
        value = ",".join(str(x) for x in value)
    argv.extend(["--" + key, str(value)])
if dry_run == "1":
    print("Resume", logdir, "to total dynamic steps", steps)
else:
    main(argv)
