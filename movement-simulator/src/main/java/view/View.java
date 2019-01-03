package view;

import abstractClasses.AbstractPlane;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.stage.Screen;
import javafx.stage.Stage;

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

    @Override
    public void setPlaneSize(double width, double height) {
        plane.setBackgroundSize(width, height);
        double wantedWidth = Screen.getPrimary().getBounds().getWidth() - 2 * Config.PLANE_MARGIN;
        double wantedHeight = Screen.getPrimary().getBounds().getHeight() - 2 * Config.PLANE_MARGIN;
        plane.setLayoutX(wantedWidth / 2 - width / 2 + Config.PLANE_MARGIN);
        plane.setLayoutY(wantedHeight / 2 - height / 2 + Config.PLANE_MARGIN);
        double scale = Double.min(wantedWidth / width,
                wantedHeight / height);
        plane.setScaleX(scale);
        plane.setScaleY(scale);
    }
}
