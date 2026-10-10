"""Build command arguments for numerical integration fixtures."""


def experiment_args(**values):
    args = []
    for key, value in values.items():
        if value is None:
            continue
        args.append("--" + key.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            args.extend(str(v) for v in value)
        else:
            args.append(str(value))
    return args
