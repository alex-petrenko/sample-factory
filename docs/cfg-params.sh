#!/bin/bash 

FILE=docs/02-configuration/cfg-params.md

cat << EOF > $FILE
# Full Parameter Reference

The command line arguments / config parameters for training using Sample Factory can be found by running your training script with the \`--help\` flag.
The list of config parameters below was obtained from running \`python -m sf_examples.train_gym_env --env=CartPole-v1 --help\`. These params can be used in any environment.
Other environments may have other custom params than can also be viewed with \`--help\` flag when running the environment-specific training script.

\`\`\`
EOF

python -m sf_examples.train_gym_env --env=CartPole-v1 --help >> $FILE

echo \`\`\` >> $FILE