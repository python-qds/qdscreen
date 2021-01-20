This is a reminder on how to grant travis the rights to deploy your site on github pages, python package to pypi and release files to github

# PREREQUISITE 

## Install docker desktop 20+

TODO link

## Test that the travis-cli image works

Open a bash prompt (on windows, use git bash. See below if you run into trouble)

```bash
> docker run -it --rm -v $(pwd):/project --entrypoint=/bin/sh skandyla/travis-cli
```

If it works this should open a `/project #` prompt. You can now exit, we'll come back to this later:

```bash
/project # exit
```

## Troubleshooting on Windows

 - if you see a message "" you are probably running docker from a legacy `cmd.exe`. [This should fix your issue](https://github.com/docker/for-win/issues/9770#issuecomment-745106453).

 - if you see a message `stat C:/Program Files/Git/usr/bin/sh: no such file or directory: unknown.` or `not a tty` this is an issue that can be solved by [using winpty](https://github.com/borekb/docker-path-workaround). Warning: the path to your actual winpty should be directly put in the script, as follows:

```
#!/bin/bash
"C:\winpty-0.4.3-msys2-2.7.0-x64\bin\winpty.exe" "docker.exe" "$@"
```

 - Finally note that there are actually two git bash executables ; you should probably not need the second one but just in case.


# Generating the access keys for travis

## To deploy a site on gh-pages using `mkdocs gh-deploy` (or for any `git push` operation)

Generate an asymetric security key (public + private):

 * On windows: open git bash (not windows cmd)
 * Execute the following but **DO NOT provide any passphrase when prompted (simply press <enter>)**

```bash
ssh-keygen -t rsa -b 4096 -C "<your_github_email_address>" -f ci_tools/github_travis_rsa
```

On the github repository page, `Settings > Deploy Keys > Add deploy key > add` the PUBLIC generated key (the file `ci_tools/github_travis_rsa.pub`) with write access


Use travis CLI to encrypt your PRIVATE key:

```bash
> cd <your_project_root>
> docker run -it --rm -v $(pwd):/project --entrypoint=/bin/sh skandyla/travis-cli
/project # travis login --com --github-token xxxxxxxxxxxxxxxxx
/project # travis whoami --com
/project # travis encrypt-file --com -r <git-username>/<repo-name> ci_tools/github_travis_rsa   (DO NOT USE --add option since it will remove all comments in your travis.yml file!)
```

Follow the instructions on screen :
- copy the line starting with `openssl ...` to your `travis.yml` file. 
- modify the relative path to the generated file by adding 'ci_tools/' in front of 'github_travis_rsa_...enc'.
- git add the generated file 'github_travis_rsa_...enc' but DO NOT ADD the private key

Note: if you find bug 'no implicit conversion of nil intro String' as mentioned [here](https://github.com/travis-ci/travis.rb/issues/190#issuecomment-377823703), [here](https://github.com/travis-ci/travis.rb/issues/585#issuecomment-374307229) and especially [here](https://github.com/travis-ci/travis.rb/issues/586) it can either be a network proxy error (check that http_proxy is not set...) or a ruby/travis cli version issue. Or worse: an openssl version issue (you check check with wireshark). Best is to build the docker container locally to be sure to get the latest version of ruby and travis cli.

source: 
   * https://djw8605.github.io/2017/02/08/deploying-docs-on-github-with-travisci/ (rejecting https://docs.travis-ci.com/user/deployment/pages/ as this would grant full access to travis)
   * https://docs.travis-ci.com/user/encrypting-files/ 
   * https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

## To deploy python wheels on PyPi

Similar procedure to encrypt the PyPi password for deployments:

```bash
> (same as above)
/project # travis encrypt --com -r <git-username>/<repo-name> <pypi_password>
```
Copy the resulting string in the `travis.yml` file under deploy > provider: pypi > password > secure

source: https://docs.travis-ci.com/user/deployment/pypi/


## To deploy file releases on github

Similar procedure to encrypt the OAuth password for github releases. **WARNING** unlike 'travis encrypt', this WILL modify your `travis.yml` file. Therefore you should make a backup of it beforehand, and then execute this command with the '--force' option.

```bash
> (same as above)
/project # travis setup releases --com
```

Copy the string in the `travis.yml` file under deploy > provider: releases > api-key > secure

source: https://docs.travis-ci.com/user/deployment/releases/
