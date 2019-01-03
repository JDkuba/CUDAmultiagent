package view.drawable;

import javafx.scene.Group;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import utility.Position;

public class Agent extends Group {
    private Circle agentShape;
    public Agent(Position position, double radius) {
        super();
        initializeAgentShape(radius);
        this.getChildren().add(agentShape);
        this.setTranslateX(position.getX());
        this.setTranslateY(position.getY());
    }

    public Agent(double translateX, double translateY, double radius) {
        super();
        initializeAgentShape(radius);
        this.getChildren().add(agentShape);
        this.setTranslateX(translateX);
        this.setTranslateY(translateY);
    }

    public void setColor(Color color) {
        agentShape.setFill(color);
    }

    public void setRadius(double radius) {
        agentShape.setRadius(radius);
    }

    private void initializeAgentShape(double radius) {
        agentShape = new Circle(radius);
    }
}
