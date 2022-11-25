# Doc Contribution

* Clone the repo. You should be in the root folder containing ‘docs’, ‘mkdocs.yml’ config file, and ‘docs.yml’ github actions file.
    
* Install dev dependencies (includes `mkdocs` deps):

```bash
pip install -e .[dev]
```

* Serve the website locally

```bash
mkdocs docs-serve
```

you should see the website on your localhost port now.

* Modify or create markdown files
- modify / create your markdown files in ‘docs’ folder.
- add your markdown path in the ‘nav’ section of ‘mkdocs.yml’.

 Example folder-yml correspondence:
 
<img src="https://user-images.githubusercontent.com/30235642/176805054-9d5f1c24-b8b6-49df-90c3-039acb741af3.png" alt="docs" width="300" height="300"/>
<img src="https://user-images.githubusercontent.com/30235642/176805061-3da4b3b6-9a18-4d87-9d06-044dc863b6e4.png" alt="yml" width="400" height="300"/>

* Commit and push your changes to remote repo. Github actions will automatically push your changes to your github pages website.
