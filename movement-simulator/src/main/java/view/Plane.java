package view;

import abstractClasses.AbstractPlane;
import utility.Position;
import view.drawable.Agent;

import java.util.ArrayList;
import java.util.List;

class Plane extends AbstractPlane {
    private List<Agent> agents;
    Plane() {
        super();
        agents = new ArrayList<>();
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
