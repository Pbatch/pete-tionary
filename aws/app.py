#!/usr/bin/env python3

from aws_cdk import core
from pictionary.pictionary_stack import PictionaryStack

env = {'region': 'eu-west-1',
       'account': '068121675185'}
app = core.App()
PictionaryStack(app, "pictionary", env=env)

app.synth()