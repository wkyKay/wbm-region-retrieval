"""Independent implementations for proposed, handcrafted and supervised methods.

The unified experiment runner dispatches training and retrieval to these modules.
Reusable encoders, protocol handling, matching and evaluation stay in the parent
package; importing this package does not import Torch or launch any computation.
"""
