#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/sphinx-doc/sphinx /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 9ed054279aeffd5b1d0642e2d24a8800389de29f
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
sed -i 's/pytest/pytest -rA/' tox.ini
sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py
sed -i 's/sphinxcontrib-applehelp/sphinxcontrib-applehelp<=1.0.7/' setup.py
sed -i 's/sphinxcontrib-devhelp/sphinxcontrib-devhelp<=1.0.5/' setup.py
sed -i 's/sphinxcontrib-qthelp/sphinxcontrib-qthelp<=1.0.6/' setup.py
sed -i 's/alabaster>=0.7,<0.8/alabaster>=0.7,<0.7.12/' setup.py
sed -i "s/'packaging',/'packaging', 'markupsafe<=2.0.1',/" setup.py
sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py
sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py
python -m pip install -e .[test]
