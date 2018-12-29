package drawable;

import javafx.scene.Group;
import javafx.scene.shape.Circle;

public class Agent extends Group {
    private Circle agentShape;
    public Agent(double translateX, double translateY, double size) {
        super();
        initializeAgentShape(size);
        this.getChildren().add(agentShape);
        this.setTranslateX(translateX);
        this.setTranslateY(translateY);
    }

    private void initializeAgentShape(double size) {
        agentShape = new Circle(size / 2);
    }
}
