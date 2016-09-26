##R Matrix Axis Function:

matrix.axes <- function(data) {
# Do the rows, las=2 for text perpendicular to the axis
x <- (1:dim(data)[1] - 1) / (dim(data)[1] - 1);
axis(side=1, at=x, labels=rownames(data), las=2);
# Do the columns
x <- (1:dim(data)[2] - 1) / (dim(data)[2] - 1);
axis(side=2, at=x, labels=colnames(data), las=2);
# Add a solid black grid
grid(nx=(dim(data)[1]-1), ny=(dim(data)[2]-1), col="black", lty="solid");
}