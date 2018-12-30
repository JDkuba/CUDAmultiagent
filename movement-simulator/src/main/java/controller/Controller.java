package controller;

import abstractClasses.AbstractPlane;
import javafx.stage.Stage;
import utility.Position;
import view.View;

import java.util.ArrayList;

public class Controller {
    private interfaces.View view;
    public Controller(Stage primaryStage) {
        this.view = new View(primaryStage);
    }

    public void run() {
        AbstractPlane plane = view.getPlane();
        ArrayList<Position> agents = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            agents.add(new Position(i*30, i*30));
        }
        plane.addAgents(agents);

        plane.setOnMouseClicked(event -> {
            for (int i = 1; i <= 10; i++) {
                agents.get(i-1).setX(plane.getWidth()-i*30);
                agents.get(i-1).setY(plane.getHeight()-i*30);
            }
            plane.moveAgents(agents);
        });
    }
}
