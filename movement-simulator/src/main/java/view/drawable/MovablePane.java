package view.drawable;

import javafx.event.Event;
import javafx.event.EventHandler;
import javafx.event.EventType;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Pane;

public class MovablePane extends Pane {
    private final double ZOOM_SCALE = 1.1;
    private final double MAX_SCALE = 10;
    private final double MIN_SCALE = 0.1;
    private double orgMousePressedX;
    private double orgMousePressedY;

    public MovablePane() {
        super();
        setOrgMousePressed();
        setOnScroll();
    }

    private void setOrgMouseDragged() {
        super.setOnMouseDragged(event -> {
            //      setManaged(false);
            //     event.consume();
            if (event.isPrimaryButtonDown()) {
                this.setTranslateX((event.getX() - orgMousePressedX) * this.getScaleX() + this.getTranslateX());
                this.setTranslateY((event.getY() - orgMousePressedY) * this.getScaleY() + this.getTranslateY());
            }
        });
    }

    private void setOrgMousePressed() {
        super.setOnMousePressed(event -> {
            setOrgMouseDragged();
            orgMousePressedX = event.getX();
            orgMousePressedY = event.getY();
        });
    }

    private double currentLeftSideX() {
        return (this.getLayoutX() + (this.getWidth() / 2) -
                (this.getWidth() * this.getScaleX()) / 2 + this.getTranslateX());
    }

    private double currentTopSideY() {
        return (this.getLayoutY() + (this.getHeight() / 2) -
                (this.getHeight() * this.getScaleY()) / 2 + this.getTranslateY());
    }

    private double scrollEventSetScale(ScrollEvent event) {
        double scaleFactor = (event.getDeltaY() > 0) ? ZOOM_SCALE : 1 / ZOOM_SCALE;
        if (this.getScaleX() * scaleFactor > MAX_SCALE) {
            scaleFactor = MAX_SCALE / this.getScaleX();
        } else if (this.getScaleX() * scaleFactor < MIN_SCALE) {
            scaleFactor = MIN_SCALE / this.getScaleX();
        }

        this.setScaleX(this.getScaleX() * scaleFactor);
        this.setScaleY(this.getScaleY() * scaleFactor);

        return scaleFactor;
    }

    private void setOnScroll() {
        super.setOnScroll(event -> {
            //       event.consume();
            if (event.getDeltaY() == 0) {
                return;
            }
            double previousLeftSideX = currentLeftSideX();
            double previousTopSideY = currentTopSideY();
            double orgDistScrollX = event.getSceneX() - previousLeftSideX;
            double orgDistScrollY = event.getSceneY() - previousTopSideY;

            double scaleFactor = scrollEventSetScale(event);

            double currLeftSideX = currentLeftSideX();
            double currTopSideY = currentTopSideY();
            this.setTranslateX(this.getTranslateX() + (event.getSceneX() - orgDistScrollX * scaleFactor) - currLeftSideX);
            this.setTranslateY(this.getTranslateY() + (event.getSceneY() - orgDistScrollY * scaleFactor) - currTopSideY);
        });
    }

    public void lockPane() {
        super.setMouseTransparent(true);
        super.setOnMousePressed(null);
        super.setOnMouseDragged(null);
        super.setOnScroll(null);
    }

    public void unlockPane() {
        setOrgMousePressed();
        setOnScroll();
        super.setMouseTransparent(false);
    }
}