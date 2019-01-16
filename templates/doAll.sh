if [[ $# < 2 ]]; then
    echo "signature is: --testType n_agents"
fi

PROJECT_DIR="/home/z1143051/cuda/MultiAgentSystem"

echo "TRANSFERRING..."
./updateOnMiracle.sh > /dev/null
echo "MAKING..."
if ! ssh miracle "cd $PROJECT_DIR; make" > /dev/null 2> err 
then
    echo "MAKING ERROR!"
    cat err
    exit
fi
rm err
echo "RUNNING..."
if ! ssh miracle "cd $PROJECT_DIR; ./bin/programd $@ < simpleTest.in; "
then
    echo "RUNNING ERROR!"
    exit
fi
echo "COMPRESSING..."
if ! ssh miracle "cd $PROJECT_DIR; tar -jcvf data.tar.bz2 *.out; rm *.out" > /dev/null
then
    echo "COMPRESSING ERROR!"
    exit
fi
echo "DOWNLOADING..."
./getDataFromMiracle.sh > /dev/null
echo "ANIMATING..."
./runAnimation.sh > /dev/null
echo "FINISHED!"




