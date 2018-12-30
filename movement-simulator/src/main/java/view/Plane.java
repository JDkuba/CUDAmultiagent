package view;

import abstractClasses.AbstractPlane;
import javafx.animation.ParallelTransition;
import javafx.animation.TranslateTransition;
import javafx.scene.shape.Rectangle;
import javafx.util.Duration;
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
            Agent agent = new Agent(position, Config.AGENT_SIZE);
            agent.setColor(Config.STANDARD_AGENT_COLOR);
            agents.add(agent);
            this.getChildren().add(agent);
        }
    }

    @Override
    public void moveAgents(List<Position> positions) {
        ParallelTransition parallelTransition = new ParallelTransition();
        for (int i = 0; i < positions.size(); i++) {
            TranslateTransition translateTransition = new TranslateTransition(
                    Config.AGENT_TRANSITION_DURATION, agents.get(i));
            translateTransition.setToX(positions.get(i).getX());
            translateTransition.setToY(positions.get(i).getY());
            parallelTransition.getChildren().add(translateTransition);
        }
        parallelTransition.playFromStart();
    }
}
