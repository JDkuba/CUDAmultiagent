package view;

import abstractClasses.AbstractPlane;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.stage.Screen;
import javafx.stage.Stage;
import utility.Position;
import view.drawable.Agent;

import java.util.ArrayList;
import java.util.List;

public class View implements interfaces.View {
    private Stage primaryStage;
    private Pane root;
    private AbstractPlane plane;
    public View(Stage primaryStage) {
        this.primaryStage = primaryStage;
        initializePrimaryStage();
        initializePlane();
        this.primaryStage.show();
    }

    private void initializePlane() {
        plane = new Plane(Screen.getPrimary().getBounds().getWidth(),
                Screen.getPrimary().getBounds().getHeight());
        root.getChildren().add(plane);
    }

    private void initializePrimaryStage() {
        this.root = new Pane();
        this.primaryStage.setFullScreen(true);
        this.root.setPrefWidth(Screen.getPrimary().getBounds().getWidth());
        this.root.setPrefHeight(Screen.getPrimary().getBounds().getHeight());
        this.primaryStage.setTitle("Movement-Simulator");
        this.primaryStage.setScene(new Scene(this.root));
    }

    @Override
    public AbstractPlane getPlane() {
        return plane;
    }
}
