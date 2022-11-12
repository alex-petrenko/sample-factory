#!/bin/bash 

FILE=docs/get-started/cfg-params.md

cat << EOF > $FILE
# Configuration Parameters
## Training Parameters
The command line arguments / config parameters for training using Sample Factory can be found by running your training script with the \`--help\` flag. 
The list of config parameters below was obtained from running \`python -m sf_examples.train_gym_env --env=CartPole-v1 --help\`. These params can be used in any environment.
Other environments may have other custom params than can be viewed with \`--help\`

\`\`\`
EOF

python -m sf_examples.train_gym_env --env=CartPole-v1 --help >> $FILE

echo \`\`\` >> $FILE