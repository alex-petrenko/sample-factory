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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/757cf2b6-30c3-49e5-8cc9-41bf9f746e04/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cd5d2c75-3565-4a7e-b2bd-5b6ac61d67d1/Untitled.png)

5. commit and push your changes to remote repo. github actions will automatically push your changes to your github pages website.
