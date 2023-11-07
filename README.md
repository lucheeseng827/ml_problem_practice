# ML Problem practice

This is a repo that is a work in progress, it's used to practice ML problems and to learn how to deploy ML models in production at the same time covering basic CICD/Devops workflow and practices.




## How to run the code:


1. Fulfill the requirements
  - python 3.9 recommended
  - pip install -r requirements.txt
  - cloud(aws,gcp) cli installed and configured
  - credentials with access to bucket
  - rust installed (*optional*)
2. Run the script

```bash
python main.py
```


For rust version:
1. move the file into rust folder
2. cargo init <project-name: s3-rust-run>
3. move the rust code into the main.rs
4. cargo add neccessary dependencies(eg: tokio, rusoto_core, rusoto_s3 etc)
5. cargo build


Precommit hooks
- hooks
- https://github.com/astral-sh/ruff-pre-commit
- https://github.com/astral-sh/ruff
