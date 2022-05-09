# Design

## Explainer CLI, API

``````{tabbed} CLI
`````{panels} 
:container: +full-width
:column: col-lg-4 px-2 py-2
:card: rounded
````{dropdown} Subcommands
:open:
* explain
* visualize
````
`````
``````

``````{tabbed} API
`````{panels} 
:container: +full-width
:column: col-lg-4 px-2 py-2
:card: rounded
````{dropdown} Methods
:open:
* explain
* visualize
````
`````
``````


## Captum Attribution Hierarchy (partial)

```{mermaid}
graph TD;
    Attribution-->GradientAttribution;
    Attribution-->PerturbationAttribution;
    Attribution-->InternalAttribution;
    InternalAttribution-->LayerAttribution;
    InternalAttribution-->NeuronAttribution;
    GradientAttribution-->NeuronGradient;
    NeuronAttribution-->NeuronGradient;
    LayerAttribution-->LayerGradCam;
    GradientAttribution-->LayerGradCam;
```


```{mermaid}
stateDiagram-v2
    state Email_Returned <<choice>>
    state Phone_Call_Choice <<choice>>
    state Interview_Choice <<choice>>
    state Offer_Response <<choice>>
    [*] --> Needs_Screening
    Needs_Screening --> Email_Sent
    Email_Sent --> Email_Returned
    Email_Returned --> Needs_Phone_Call: ✅ reply
    Email_Returned --> [*]: ❎ declines
    Email_Returned --> [*] : ❎ no reply
    Needs_Phone_Call --> Phone_Call_Scheduled
    Phone_Call_Scheduled --> Phone_Call
    Phone_Call --> Phone_Call_Choice
    Phone_Call_Choice --> Needs_Interview: ✅ passes
    Phone_Call_Choice --> [*]: ❎ fails
    Needs_Interview --> Interview_Scheduled
    Interview_Scheduled --> Interview
    Interview --> Interview_Choice
    Interview_Choice --> Extend_Offer: ✅ passes
    Interview_Choice --> Reject: ❎ fails
    Interview_Choice --> Hold: ❎ undecided
    Reject --> [*]
    Hold --> [*]
    Extend_Offer --> Offer_Extended
    Offer_Extended --> Offer_Response
    Offer_Response --> Offer_Accepted: ✅ accepts
    Offer_Response --> Pending_Visa: ✅ accepts
    Offer_Response --> Declined: ❎ declines
    Offer_Accepted --> [*]
    Pending_Visa --> [*]
    Declined --> [*]
```
