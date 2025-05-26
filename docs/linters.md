### 1. check code with three linters
- pylint
```
pylint src/ --rcfile=.pylintrc # should show nothing
```

- flake8
```
flake8 src/ --config=.flake8 # should show nothing
```

- bandit
```
bandit -r src/ # should just show low risk issues
```

### 2. check custom rules
```
cp pylint_custom/warning_example.py src/
```
```
pylint src/ --rcfile=.pylintrc # should show custom warnings
```
```
rm src/warning_example.py
```
