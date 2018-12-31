package view.drawable;

import javafx.scene.Group;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import utility.Position;

public class Agent extends Group {
    private Circle agentShape;
    public Agent(Position position, double size) {
        super();
        initializeAgentShape(size);
        this.getChildren().add(agentShape);
        this.setTranslateX(position.getX());
        this.setTranslateY(position.getY());
    }

    public Agent(double translateX, double translateY, double size) {
        super();
        initializeAgentShape(size);
        this.getChildren().add(agentShape);
        this.setTranslateX(translateX);
        this.setTranslateY(translateY);
    }

    public void setColor(Color color) {
        agentShape.setFill(color);
    }

    public void setSize(double size) {
        agentShape.setRadius(size/2);
    }

    private void initializeAgentShape(double size) {
        agentShape = new Circle(size / 2);
    }


}
