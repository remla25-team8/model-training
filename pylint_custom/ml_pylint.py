from pylint.checkers import BaseChecker
from astroid import nodes

is_verbose = False  # Set to True for verbose output

class MLSpecificSmellsChecker(BaseChecker):
    name = 'ml-smells'
    priority = -1
    msgs = {
        "W9999": (
            "Hardcoded path detected. Use os.path.join() for reproducibility.",
            "hardcoded-path",
            "Avoid hardcoded paths in ML code.",
        ),
        "W9998": (
            "Missing or None random_state in train_test_split(). Set it for reproducibility.",
            "missing-random-state",
            "Always set a non-None random_state for reproducible train-test splits.",
        ),
    }

    def nodes(self):
        # Explicitly declare which node types this checker is interested in
        return [nodes.Const, nodes.Call]

    def visit_const(self, node):
        """Checks for hardcoded paths."""
        if isinstance(node.value, str):
            print(f"Checking constant value: {node.value}") if is_verbose else None
            if "\\" in node.value or node.value.startswith("/"):
                if not (node.value.startswith("./") or node.value.startswith("../")):
                    self.add_message("hardcoded-path", node=node)

    def visit_call(self, node):
        """Checks for missing or None random_state in train_test_split."""
        if (
            isinstance(node.func, nodes.Name) and node.func.name == "train_test_split"
        ) or (
            isinstance(node.func, nodes.Attribute)
            and node.func.attrname == "train_test_split"
        ):
            keywords = {kw.arg: kw.value for kw in node.keywords or []}

            if "random_state" not in keywords:
                # random_state is missing
                self.add_message("missing-random-state", node=node)
            else:
                # Check if it's explicitly set to None
                random_state_val = keywords["random_state"]
                if isinstance(random_state_val, nodes.Const) and random_state_val.value is None:
                    self.add_message("missing-random-state", node=node)


def register(linter):
    print("Registering ml-smells plugin") if is_verbose else None
    linter.register_checker(MLSpecificSmellsChecker(linter))
