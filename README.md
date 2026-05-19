# newton cloth demo

Look at the [newton](https://github.com/newton-physics/newton) github page and follow instructions for setting up dependencies.
Afterward, add package to your PYTHONPATH.

You can then run a teleop script for cloth manipulation with the dual arm piper-x:
```bash
python cloth_teleop --grasp-mode {constraint, normal} --cloth-asset {grid, shirt}
```

## Teleop controls

| Key | Action |
| --- | --- |
| `T` | Switch teleop between the left and right arm. |
| `I` / `K` | Move the active end-effector along world X. |
| `J` / `L` | Move the active end-effector along world Y. |
| `U` / `O` | Move the active end-effector along world Z. |
| `R` / `F` | Pitch the active end-effector. |
| `G` | Toggle the active gripper open/closed. |

## Options

| Option | Description |
| --- | --- |
| `--cloth-asset grid` | Use Newton's square cloth mesh from `example_cloth_twist.py`. |
| `--cloth-asset shirt` | Use Newton's unisex shirt mesh as the cloth. |
| `--grasp-mode constraint` | Attach nearby cloth particles when the gripper closes. |
| `--grasp-mode normal` | Only actuate the gripper, with no explicit cloth attachment. |

## Examples

### --cloth-asset grid
![Cloth teleop example](images/cloth_example.png)

### --cloth-asset shirt
![Shirt teleop example](images/shirt_example.png)
