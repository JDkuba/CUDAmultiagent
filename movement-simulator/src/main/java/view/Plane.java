package view;

import abstractClasses.AbstractPlane;
import javafx.scene.shape.Rectangle;
import utility.Position;
import view.drawable.Agent;

import java.util.ArrayList;
import java.util.List;

class Plane extends AbstractPlane {
    private List<Agent> agents;
    private Rectangle background;
    Plane(double width, double height) {
        super();
        initializeBackground(width, height);
        agents = new ArrayList<>();
    }

    private void initializeBackground(double width, double height) {
        background = new Rectangle(width, height);
        background.setFill(Config.PLANE_BACKGROUND_COLOR);
        this.getChildren().add(background);
    }

    @Override
    public void addAgents(List<Position> positions) {
        for (Position position : positions) {
            agents.add(new Agent(position, Config.AGENT_SIZE));
            this.getChildren().add(agents.get(agents.size()-1));
        }
    }

    @Override
    public void moveAgents(List<Position> positions) {

    }
}
