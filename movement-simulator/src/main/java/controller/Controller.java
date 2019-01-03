package controller;

import abstractClasses.AbstractPlane;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import cudaUtils.CudaSceneDataBox;
import cudaUtils.CudaSceneMetadata;
import javafx.animation.Transition;
import javafx.stage.Stage;
import utility.Position;
import view.View;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Controller {
    private interfaces.View view;

    public Controller(Stage primaryStage) {
        this.view = new View(primaryStage);
    }

    public void setAnimation(CudaSceneDataBox dataBox) throws IOException {
        AbstractPlane plane = view.getPlane();

        plane.addAgents(dataBox.getNextPositionsList());

        CudaSceneMetadata metaData = dataBox.getCudaSceneMetadata();
        plane.setAgentsSize(metaData.getAgentRadius());


        List<List<Position>> positionsLists = new ArrayList<>();
        for (int i = 2; i <= metaData.getGenerationsNumber(); i++) {
            positionsLists.add(dataBox.getNextPositionsList());

        }
        addAnimationInPackages(20, positionsLists, plane);

        plane.setOnMouseClicked(event -> plane.playAgentSimulationFromStart());
    }

    private void addAnimationInPackages(int packageSize, List<List<Position>> positionsLists, AbstractPlane plane) {
        for (int i = 0, j = packageSize; j <= positionsLists.size(); i += packageSize, j += packageSize) {
            plane.addAgentsPathTranslation(positionsLists.subList(i, j));
        }
    }
}
