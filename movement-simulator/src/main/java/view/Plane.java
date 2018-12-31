package view;

import abstractClasses.AbstractPlane;
import javafx.animation.*;
import javafx.scene.shape.LineTo;
import javafx.scene.shape.MoveTo;
import javafx.scene.shape.Path;
import javafx.scene.shape.Rectangle;
import javafx.util.Duration;
import utility.Position;
import view.drawable.Agent;

import java.util.ArrayList;
import java.util.List;

class Plane extends AbstractPlane {
    private List<Agent> agents;
    private Rectangle background;
    private double agentSize;
    Plane(double width, double height) {
        super();
        initializeBackground(width, height);
        this.agents = new ArrayList<>();
        this.agentSize = Config.AGENT_SIZE;
    }

    private void initializeBackground(double width, double height) {
        background = new Rectangle(width, height);
        background.setFill(Config.PLANE_BACKGROUND_COLOR);
        this.getChildren().add(background);
    }

    @Override
    public void addAgents(List<Position> positions) {
        for (Position position : positions) {
            Agent agent = new Agent(position, agentSize);
            agent.setColor(Config.STANDARD_AGENT_COLOR);
            agents.add(agent);
            this.getChildren().add(agent);
        }
    }

    @Override
    public void setAgentsPositions(List<Position> positions) {
        for (int i = 0; i < positions.size(); i++) {
            agents.get(i).setTranslateX(positions.get(i).getX());
            agents.get(i).setTranslateY(positions.get(i).getY());
        }
    }

    @Override
    public void setAgentsSize(double size) {
        agentSize = size;
        for (Agent agent : agents) {
            agent.setSize(size);
        }
    }

    @Override
    public void transitionTranslateAgents(List<Position> positions) {
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

    @Override
    public Transition getPathTranslateAgents(List<List<Position>> agentMovements) {
        ParallelTransition parallelTransition = new ParallelTransition();
        int totalNumberOfTranslations = 0;
        for (int i = 1; i < agentMovements.size(); i++) {
            System.out.println("making animation number " + i + " out of " + (agentMovements.size()-1));
            SequentialTransition sequentialTransition = new SequentialTransition();
            for (Position position : agentMovements.get(i)) {
                TranslateTransition translateTransition = new TranslateTransition(Duration.millis(20), agents.get(i));
                translateTransition.setToX(position.getX());
                translateTransition.setToY(position.getY());
                totalNumberOfTranslations++;
                sequentialTransition.getChildren().add(translateTransition);
            }
            parallelTransition.getChildren().add(sequentialTransition);
            /*Path path = new Path();
            Position prevPos = new Position(agents.get(i).getTranslateX(), agents.get(i).getTranslateY());
            path.getElements().add(new MoveTo(prevPos.getX(), prevPos.getY()));
            for (Position position : agentMovements.get(i)) {
                if (prevPos.equals(position)) {
                    path.getElements().add(new MoveTo(position.getX(), position.getY()));
                } else {
                    path.getElements().add(new LineTo(position.getX(), position.getY()));
                }
            }
            PathTransition pathTransition = new PathTransition(
                    Duration.millis(20 * agentMovements.get(i).size()), path, agents.get(i));
            parallelTransition.getChildren().add(pathTransition);*/
        }
        System.out.println("Total number of translations: " + totalNumberOfTranslations);
        return parallelTransition;
    }
}
