Tetris_RL_player
====================

code written by Chen Liang and Yetong Zhang.

To run the code, first install the requirements by running:

::

    cd RL
    pip3 install -r requirements.txt

To run Java server:

::

    cd java
    ./gradlew run


To run validation experiment using trained model:

::

    cd RL
    python3 run_java.py --experiment [exp1] # you can replace exp1 with the name of the experiment

Reference
====================

[1] Python Java Interface: https://github.com/bartdag/py4j-smallbench-example
