# Doc Contribution

## workflows
1. clone the target repo 
    
    *It should contain a ‘docs’ folder, a ‘mkdocs.yml’ config file, a ‘docs.yml’ github actions file.*
    
2. install common dependencies

```bash
pip install mkdocs-material
pip install mkdocs-minify-plugin
pip install mkdocs-redirects
pip install mkdocs-git-revision-date-localized-plugin
pip install mkdocs-git-committers-plugin-2
pip install mkdocs-git-authors-plugin
```

3. serve the website locally

```bash
mkdocs serve
```
you should see the website on your localhost port now.

4. modify or create markdown files
- modify / create your markdown files in ‘docs’ folder.
- add your markdown path in the ‘nav’ section of ‘mkdocs.yml’(at the bottom).

 Example folder-yml correspondence:
 
<img src="https://user-images.githubusercontent.com/30235642/176805054-9d5f1c24-b8b6-49df-90c3-039acb741af3.png" alt="docs" width="300" height="300"/>
<img src="https://user-images.githubusercontent.com/30235642/176805061-3da4b3b6-9a18-4d87-9d06-044dc863b6e4.png" alt="yml" width="400" height="300"/>

5. commit and push your changes to remote repo. github actions will automatically push your changes to your github pages website.
