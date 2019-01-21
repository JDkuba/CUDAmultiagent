package controller;

import abstractClasses.AbstractPlane;
import cudaUtils.CudaSceneDataBox;
import cudaUtils.CudaSceneMetadata;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.stage.Stage;
import javafx.util.Duration;
import utility.Position;
import view.View;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Controller {
    private interfaces.View view;

    public Controller(Stage primaryStage) {
        this.view = new View(primaryStage);
    }

    public void setAnimation(CudaSceneDataBox dataBox) throws IOException {
        AbstractPlane plane = view.getPlane();

        List<Position> startingPositions = dataBox.getNextPositionsList();
        plane.addAgents(startingPositions);

        CudaSceneMetadata metaData = dataBox.getCudaSceneMetadata();

        view.setPlaneSize(metaData.getBoardX(), metaData.getBoardY());
        plane.setAgentsRadius(metaData.getAgentRadius());
        plane.setFrameDuration(Duration.millis(10));

        List<List<Position>> positionsLists = new ArrayList<>();
        for (int i = 2; i <= metaData.getGenerationsNumber(); i++) {
            positionsLists.add(dataBox.getNextPositionsList());

        }
        addAnimationInPackages(20, positionsLists, plane);

        plane.setFocusTraversable(true);
        plane.setOnKeyReleased(keyEvent -> {
            if (keyEvent.getCode().compareTo(KeyCode.R) == 0) {
                plane.playAgentSimulationFromStart();
                plane.stopAgentSimulation();
                plane.setAgentsPositions(startingPositions);
            }

            if (keyEvent.getCode().compareTo(KeyCode.SPACE) == 0) {
                plane.changeAgentSimulationPlayStatus();
            }
        });
    }

    private void addAnimationInPackages(int packageSize, List<List<Position>> positionsLists, AbstractPlane plane) {
        for (int i = 0, j = packageSize; j <= positionsLists.size(); i += packageSize, j += packageSize) {
            plane.addAgentsPathTranslation(positionsLists.subList(i, j));
        }
    }
}
