#!/bin/sh

if [ -x ./build/bin/demo ]
then
	./build/bin/demo
else
	>&2 echo "No executable found."
fi