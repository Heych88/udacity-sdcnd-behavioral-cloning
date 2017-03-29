# udacity-sdcnd-behavioral-cloning
Udacity Project 3 Behavioral-Cloning

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites

Required python packages

```
numpy
opencv3.2
sklearn
Keras
Tensorflow
Matplotlib
```

The required simulator software can be found and installed by following the instructions under the [Udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim) Project on github.


## Running the Model

### Collecting Data

Once the simulator has been downloaded, Start the simulator by navigating to the simulators saved directory and double click on the simulator version for the computer system you have. 

For 32-bit Linux:  `<directory>/linux_sim/linux_sim.x86`

For 64-bit Linux:  `<directory>/linux_sim/linux_sim.x86_64`

Chose the graphics properties best suted for your operating system and then click 'ok'

In the simulator, select the track you wish to collect data from and then click on 'TRAINING MODE'.

Drive around the track by using the arrow keys of the keyboard or using the mouse to steer by holding down the left click.

Record driving data by clicking on the 'RECORD' button and selecting the output folder location. 
It is recomended to create a folder called data in the same directory location as the git repository.
Click on the folder to save the data and press 'Select'.
Click 'RECORD' once again to record the driving.

To finish collecting data, click the 'RECORD' button to finish.

### Training a Model
Once data has been collected. Open a terminal window and navigate to the location of the saved repository.
Train the model by typing the following into terminal

`
python model.py
`

This will save the trained model as `model.h5`

### Atonomous Mode

In the simulator, select the track you wish to test on and then click on 'AUTONOMOUS MODE'.
In a terminal window, navigate the the location of downloaded git repository.
Start the model by typing the following into terminal

`
python drive.py model.h5
`

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
