package utility;

public class Position {
    private double x, y;
    public Position(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public double getX() {
        return x;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Position) {
            return ((Position) o).x == this.x && ((Position) o).y == this.y;
        } else {
            return super.equals(o);
        }
    }
}
