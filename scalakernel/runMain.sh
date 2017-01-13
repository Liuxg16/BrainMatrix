#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../"; pwd)
CLASS_PATH=$MXNET_ROOT/scalakernel/target/*:$MXNET_ROOT/scalakernel/target/classes/*:$MXNET_ROOT/scalakernel/target/test-classes/*
INPUT_IMG=$1 
echo $MXNET_ROOT
mvn exec:java -e -Dexec.mainClass="thu.brainmatrix.Main" 

# mvn exec:java -Dexec.mainClass="com.vineetmanohar.module.Main" -Dexec.args="arg0 arg1 arg2"  -e
