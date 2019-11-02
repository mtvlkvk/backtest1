from all_imports import go


def draw(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.values,
        name='Curve',
        line_color='red',
        opacity=0.8
    ))

    fig.show()
